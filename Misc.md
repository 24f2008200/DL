I derived it by turning the momentum update into a recurrence on the error and then asking what kind of sequence it produces.

**Step 1: shift the problem so the minimum is at 0.**
Let e_t = w_t − 5. Then L = e², and the gradient is 2e_t.

**Step 2: write the update (your course's convention).**
- u_t = β·u_{t−1} + 2e_{t−1}
- e_t = e_{t−1} − η·u_t

**Step 3: eliminate u.**
From the second line, e_t − e_{t−1} = −η·u_t. One step earlier, e_{t−1} − e_{t−2} = −η·u_{t−1}, so u_{t−1} = −(e_{t−1} − e_{t−2})/η.

Substitute u_t = β·u_{t−1} + 2e_{t−1} into e_t − e_{t−1} = −η·u_t:

e_t − e_{t−1} = β(e_{t−1} − e_{t−2}) − 2η·e_{t−1}

Rearranging:

**e_t = (1 + β − 2η)·e_{t−1} − β·e_{t−2}**

With η = 0.04, 2η = 0.08.

**Step 4: get the characteristic equation.**
For a linear recurrence, try e_t = λ^t. Plugging in and dividing by λ^(t−2) gives

λ² = (1 + β − 0.08)·λ − β, which is λ² − (1 + β − 0.08)λ + β = 0.

**Why it tells you the behavior:**
- The two roots λ₁, λ₂ control how e_t evolves, as e_t = c₁λ₁^t + c₂λ₂^t.
- If the roots are **real**, the error decays smoothly with no oscillation (β = 0 and β = 0.5).
- If they are **complex**, the error oscillates. The product of the roots equals β, so both have magnitude √β. That means the oscillation shrinks by a factor of √β per step: √0.9 ≈ 0.949 and √0.99 ≈ 0.995. This is why β = 0.99 barely damps.
- The discriminant (1 + β − 0.08)² − 4β decides real versus complex. It is positive for β = 0 and 0.5, and negative for β = 0.9 and 0.99.

For β = 0 the recurrence reduces to e_t = 0.92·e_{t−1}, which is plain gradient descent.

