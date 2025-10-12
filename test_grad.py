import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck

class RnnCudaFunction(Function):
    @staticmethod
    def forward(ctx, x, A, B):
        """
        Inputs:
          x : (N,1) column vector
          A : (N,N)
          B : (N,N)
        Returns: tuple (y, x2)
          y  = 0.15 * tanh( A @ x )
          x2 = x + B @ (x**2)   # x**2 is elementwise
        """
        # Ensure contiguous (helpful for some backends)
        x = x.contiguous()
        A = A.contiguous()
        B = B.contiguous()

        # forward computations
        q = x * x                 # (N,1) elementwise square
        x2 = x + B.matmul(q)      # (N,1)

        u = A.matmul(x)           # (N,1)
        tanh_u = torch.tanh(u)    # (N,1)
        y = 0.15 * tanh_u         # (N,1)

        # save for backward
        ctx.save_for_backward(x, A, B, u, tanh_u, q)

        return y, x2

    @staticmethod
    def backward(ctx, grad_y, grad_x2):
        """
        grad_y : dL/dy  (N,1)
        grad_x2: dL/dx2 (N,1)

        Returns gradients (dx, dA, dB) in the same order as forward inputs.
        """
        x, A, B, u, tanh_u, q = ctx.saved_tensors

        g = grad_y   # (N,1)
        h = grad_x2  # (N,1)

        # --- y branch ---
        r = 0.15 * (1.0 - tanh_u * tanh_u)    # ds/du (N,1)
        delta = r * g                         # elementwise (N,1)

        # dA = delta @ x^T
        dA = delta.matmul(x.t())              # (N,1) @ (1,N) -> (N,N)

        # dx contribution from y: A^T @ delta
        dx_from_y = A.t().matmul(delta)       # (N,N) @ (N,1) -> (N,1)

        # --- x2 branch ---
        # dB = h @ q^T  (outer product)
        dB = h.matmul(q.t())                  # (N,1) @ (1,N) -> (N,N)

        # dx contribution from x2:
        # dx_from_x2 = h + 2 * x * (B^T @ h)
        Bt_h = B.t().matmul(h)                # (N,1)
        dx_from_x2 = h + 2.0 * x * Bt_h       # elementwise (N,1)

        # total dx
        dx = dx_from_y + dx_from_x2           # (N,1)

        # return gradients for inputs (dx, dA, dB)
        return dx, dA, dB

class RnnCudaFunction_Recompute(Function):
    @staticmethod
    def forward(ctx, x, A, B):
        """
        Inputs:
          x : (N,1)
          A : (N,N)
          B : (N,N)
        Returns:
          y  = 0.15 * tanh(A @ x)
          x2 = x + B @ (x**2)
        """
        y = 0.15 * torch.tanh(A @ x)
        x2 = x + B @ (x * x)
        # Save only inputs (minimal memory)
        ctx.save_for_backward(x, A, B)
        return y, x2

    @staticmethod
    def backward(ctx, grad_y, grad_x2):
        # Retrieve inputs
        x, A, B = ctx.saved_tensors

        # --- Recompute intermediates ---
        u = A @ x
        tanh_u = torch.tanh(u)

        # --- Gradients ---
        r = 0.15 * (1.0 - tanh_u * tanh_u)      # derivative of tanh part
        delta = r * grad_y                      # (N,1)

        # Grad w.r.t. A, B
        dA = delta @ x.t()                      # (N,N)
        q = x * x
        dB = grad_x2 @ q.t()                    # (N,N)

        # Grad w.r.t. x
        dx = A.t() @ delta + grad_x2 + 2.0 * x * (B.t() @ grad_x2)

        return dx, dA, dB

def calc(x, A, B):
    x2 = x + B @ (x**2)
    x = A @ x
    x = 0.15 * torch.tanh(x)
    return x, x2

class RnnCuda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A, B):
        # return RnnCudaFunction.apply(x, A, B)
        return RnnCudaFunction_Recompute.apply(x, A, B)


if __name__ == "__main__":
    # Small gradcheck demo (requires double precision)
    torch.manual_seed(0)
    N = 4

    x = torch.randn(N, 1, dtype=torch.double, requires_grad=True)
    A = torch.randn(N, N, dtype=torch.double, requires_grad=True)
    B = torch.randn(N, N, dtype=torch.double, requires_grad=True)

    # gradcheck expects a function that accepts tensors and returns tuple of tensors
    test = gradcheck(RnnCudaFunction.apply, (x, A, B),
                     eps=1e-6, atol=1e-5, rtol=1e-3)
    print("gradcheck:", test)