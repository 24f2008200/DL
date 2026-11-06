I derived it by turning the momentum update into a recurrence on the error and then asking what kind of sequence it produces.

## Step 1: Shift the problem so the minimum is at 0

Let:

$$
e_t=w_t-5
$$

Then:

$$
L=e^2
$$

and the gradient is:

$$
\frac{dL}{de}=2e_t
$$

## Step 2: Write the update (your course's convention)

The momentum update equations are:

$$
u_t=\beta u_{t-1}+2e_{t-1}
$$

$$
e_t=e_{t-1}-\eta u_t
$$

## Step 3: Eliminate $u$

From the second line:

$$
e_t-e_{t-1}=-\eta u_t
$$

One step earlier:

$$
e_{t-1}-e_{t-2}=-\eta u_{t-1}
$$

Therefore:

$$
u_{t-1}=-\frac{e_{t-1}-e_{t-2}}{\eta}
$$

Substitute $u_t=\beta u_{t-1}+2e_{t-1}$ into $e_t-e_{t-1}=-\eta u_t$:

$$
e_t-e_{t-1}
\beta(e_{t-1}-e_{t-2})-2\eta e_{t-1}
$$

Rearranging:

$$
\boxed{
e_t=(1+\beta-2\eta)e_{t-1}-\beta e_{t-2}
}
$$

With $\eta=0.04$:

$$
2\eta=0.08
$$

## Step 4: Get the characteristic equation

For a linear recurrence, try:

$$
e_t=\lambda^t
$$

Plugging it into the recurrence and dividing by $\lambda^{t-2}$ gives:

$$
\lambda^2=(1+\beta-0.08)\lambda-\beta
$$

Rearranging:

$$
\boxed{
\lambda^2-(1+\beta-0.08)\lambda+\beta=0
}
$$

## Why it tells you the behavior

- The two roots $\lambda_1,\lambda_2$ control how $e_t$ evolves:

$$
e_t=c_1\lambda_1^t+c_2\lambda_2^t
$$

- If the roots are **real**, the error decays smoothly with no oscillation ($\beta=0$ and $\beta=0.5$).

- If the roots are **complex**, the error oscillates. The product of the roots equals $\beta$:

$$
\lambda_1\lambda_2=\beta
$$

Since complex roots of a real-coefficient quadratic are conjugates, both have magnitude:

$$
|\lambda_1|=|\lambda_2|=\sqrt{\beta}
$$

That means the oscillation shrinks by a factor of $\sqrt{\beta}$ per step:

$$
\sqrt{0.9}\approx0.949
$$

$$
\sqrt{0.99}\approx0.995
$$

This is why $\beta=0.99$ barely damps.

- The discriminant determines whether the roots are real or complex:

$$
\Delta=(1+\beta-0.08)^2-4\beta
$$

It is positive for $\beta=0$ and $0.5$, and negative for $\beta=0.9$ and $0.99$.

For $\beta=0$, the recurrence reduces to:

$$
e_t=0.92e_{t-1}
$$

which is plain gradient descent.
