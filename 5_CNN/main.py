import torch


x = torch.tensor(
    3.0,
    requires_grad=True
)

z = 2 * x
y = z ** 2

y.backward()


print("x:", x)
print("z:", z)
print("y:", y)

print("Gradient:", x.grad)