Let's unfold this from first principles. The mystery behind the different versions of Gradient Descent (GD) is actually quite simple:

They differ in two independent decisions:

1. How much training data do we use to calculate a gradient?
2. How do we use the current and previous gradients to update the parameters?

The first determines whether we have Batch GD, SGD or Mini-batch GD. The second gives us Momentum, RMSProp, Adam and other optimisers.

Let's build this systematically.

# 1. Start with the fundamental idea

Suppose we have a neural network with parameters:

$$ \theta=(w_1,w_2,\ldots,w_n,b_1,\ldots) $$

Our objective is to minimise the total loss over $N$ training samples.

$$ J(\theta)=\frac{1}{N}\sum\_{i=1}^{N}L_i(\theta) $$

The gradient tells us the direction of steepest increase in loss.

Therefore, to minimise loss, we move in the opposite direction:

$$ \boxed{\theta\_{\text{new}}=\theta\_{\text{old}}-\eta\nabla J(\theta)} $$

Here:

- $\eta$ = learning rate.
- $\nabla J$ = gradient of the loss.
- $\theta$ = all trainable parameters (weights and biases).

The fundamental question is: How do we calculate that gradient?

# 2. Three versions based on batch size

Imagine a dataset containing 1000 training samples.

A. Batch Gradient Descent

Batch size = 1000

All 1000 samples contribute to one gradient calculation.

One parameter update per epoch.

B. Stochastic Gradient Descent

Batch size = 1

Each sample contributes to its own gradient calculation.

1000 parameter updates per epoch.

C. Mini-batch Gradient Descent

Batch size = 100

Ten groups, each containing 100 samples.

10 parameter updates per epoch.

## A. Batch Gradient Descent

All training samples are used to calculate the gradient.

$$ g_t=\frac{1}{N}\sum\_{i=1}^{N}\nabla L_i(\theta_t) $$

Then one update:

$$ \theta\_{t+1}=\theta_t-\eta g_t $$

Characteristics:

- Exact gradient of the average training loss.
- Stable updates.
- Computationally expensive for large datasets.
- One update after processing the entire dataset.

## B. Stochastic Gradient Descent (SGD)

Here, batch size is 1.

For each training sample $i$:

$$ g_i=\nabla L_i(\theta) $$

Update immediately:

$$ \boxed{\theta\leftarrow\theta-\eta g_i} $$

Notice something important:

After updating the parameters using sample 1, we use the new parameters when calculating the gradient for sample 2.

We do not calculate all 1000 gradients using the same old parameters.

This is the key distinction.

## C. Mini-batch Gradient Descent

This is the most commonly used approach in modern neural network training.

Suppose:

- Dataset size = 1000.
- Batch size = 100.

We divide the dataset into 10 mini-batches.

For each mini-batch:

$$ g_k=\frac{1}{100}\sum\_{i\in B_k}\nabla L_i(\theta) $$

Then:

$$ \theta\leftarrow\theta-\eta g_k $$

After 10 updates, one epoch is complete.

| Method        | Batch size | Updates per epoch |
| ------------- | ---------- | ----------------- |
| Batch GD      | 1000       | 1                 |
| SGD           | 1          | 1000              |
| Mini-batch GD | 100        | 10                |

# 3. Now introduce Momentum

So far, all three methods use the current gradient to decide the next update.

But what if we also consider the direction in which we have been moving?

This is the idea behind Momentum.

Imagine a ball rolling down a valley. Even when the slope changes, the ball retains some of its velocity.

Momentum introduces this accumulated velocity into gradient descent.

The equations are:

$$ v_t=\beta v\_{t-1}+\eta g_t $$

$$ \boxed{\theta_t=\theta\_{t-1}-v_t} $$

where:

- $g_t$ = current gradient.
- $v_t$ = accumulated velocity.
- $\beta$ = momentum coefficient, usually around 0.9.
- $\eta$ = learning rate.
<img width="845" height="431" alt="image" src="https://github.com/user-attachments/assets/c59e2c8c-20b4-4bfb-bb00-b784f3c44d42" />

### What does momentum achieve?

Consider two situations:

1. When gradients consistently point in the same direction, momentum builds up and accelerates movement.
2. When gradients alternate direction, momentum can dampen the oscillations.

It can therefore help the optimiser move efficiently through narrow valleys.

An important point: Momentum can overshoot the minimum, too. It does not guarantee that every update decreases the loss.

# 4. Adaptive learning-rate methods

Momentum uses historical gradients to build velocity.

Another idea is to use historical gradients to adjust the learning rate for individual parameters.

This leads to several important optimisers.

## A. AdaGrad

AdaGrad accumulates the squares of past gradients.

$$ r_t=r\_{t-1}+g_t^2 $$

$$ \boxed{ \theta_t=\theta\_{t-1}- \frac{\eta}{\sqrt{r_t}+\epsilon}g_t } $$

The division is element-wise.

Effect:

- Parameters with large accumulated gradients receive smaller updates.
- Parameters with small accumulated gradients receive relatively larger updates.
- Learning rates continually decrease.

This can become problematic in long training runs because the accumulated squared gradients keep growing.

## B. RMSProp

RMSProp addresses the continually decreasing learning rate of AdaGrad.

Instead of accumulating all past squared gradients, it maintains an exponentially weighted moving average:

$$ r_t=\beta r\_{t-1}+(1-\beta)g_t^2 $$

Then:

$$ \boxed{ \theta_t=\theta\_{t-1}- \frac{\eta}{\sqrt{r_t}+\epsilon}g_t } $$

Old gradients gradually lose their influence.

## C. Adam (Adaptive Moment Estimation)

Adam combines two ideas:

- Momentum: tracks the average gradient.
- RMSProp: tracks the average squared gradient.

First moment:

$$ m_t=\beta_1m\_{t-1}+(1-\beta_1)g_t $$

Second moment:

$$ v_t=\beta_2v\_{t-1}+(1-\beta_2)g_t^2 $$

Because both start at zero, Adam applies bias correction:

$$ \hat m_t=\frac{m_t}{1-\beta_1^t} $$

$$ \hat v_t=\frac{v_t}{1-\beta_2^t} $$

Finally:

$$ \boxed{ \theta_t=\theta\_{t-1}- \frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t } $$

Adam is widely used because it combines momentum with adaptive parameter-wise learning rates.

# 5. Put all the versions together

| Optimiser         | Uses batch gradients? | Remembers previous gradients?            | Adaptive learning rate? |
| ----------------- | --------------------- | ---------------------------------------- | ----------------------- |
| Batch GD          | Entire dataset        | No                                       | No                      |
| SGD               | One sample            | No                                       | No                      |
| Mini-batch GD     | Subset of samples     | No                                       | No                      |
| Momentum GD       | Usually mini-batch    | Yes, velocity                            | No                      |
| Nesterov Momentum | Usually mini-batch    | Yes, look-ahead velocity                 | No                      |
| AdaGrad           | Usually mini-batch    | Yes, squared gradients                   | Yes                     |
| RMSProp           | Usually mini-batch    | Yes, moving average of squared gradients | Yes                     |
| Adam              | Usually mini-batch    | Yes, first and second moments            | Yes                     |

Notice that Batch, SGD and Mini-batch describe how gradients are estimated.

Momentum, AdaGrad, RMSProp and Adam describe how those gradients are used to update parameters.

These are not mutually exclusive categories. For instance, you can use mini-batch training with Adam.

# 6. Your important question: When does SGD update the gradient?

Let's settle this precisely.

Suppose we have 1000 samples and batch size 1.

One epoch looks like this:

## One SGD epoch

1. Sample 1

   Calculate gradient using current parameters $\theta_0$.

   Update to $\theta_1$.
2. Sample 2

   Calculate gradient using $\theta_1$.

   Update to $\theta_2$.
3. Sample 3

   Calculate gradient using $\theta_2$.

   Update to $\theta_3$.
4. Continue through sample 1000

   Each sample produces one gradient calculation and one parameter update.

1000 updates = 1 epoch

Therefore:

SGD updates the parameters after every sample, not after every epoch.

The gradient itself is recalculated at every update using the latest parameters.

For mini-batch GD, it is recalculated after every mini-batch.

For Batch GD, it is recalculated once per epoch.

One subtlety: an epoch is a complete pass through the training dataset. The training samples are commonly shuffled before each epoch, so the order can change.

# 7. Pseudocode: See the difference

SGD (batch size 1)

```
for epoch in range(epochs):    shuffle(training_data)    for x, y in training_data:        prediction = model(x)        loss = calculate_loss(prediction, y)        gradients = backpropagation(loss)        update_parameters(gradients)
```

Mini-batch Gradient Descent

```
for epoch in range(epochs):    shuffle(training_data)    for batch in make_batches(training_data, batch_size):        predictions = model(batch.x)        loss = calculate_batch_loss(predictions, batch.y)        gradients = backpropagation(loss)        update_parameters(gradients)
```

The essential difference is where the inner-loop parameter update happens.

# 8. One final distinction worth remembering

There is some ambiguity in the terminology used in textbooks and libraries.

Strictly speaking:

- SGD means batch size 1.
- Mini-batch SGD means batch size greater than 1 and smaller than the dataset.
- Batch GD means the entire dataset.

However, in many deep-learning libraries and discussions, the term SGD is used more broadly to refer to stochastic or mini-batch optimisation.

For example, PyTorch's `torch.optim.SGD` supports both mini-batch SGD and momentum.

## The complete mental model

Two independent dimensions of gradient descent

Dimension 1: Gradient estimation

How many samples contribute to the gradient?

- Batch GD
- SGD
- Mini-batch GD

Dimension 2: Parameter update strategy

How do we use the calculated gradient?

- Vanilla GD
- Momentum / Nesterov
- AdaGrad
- RMSProp
- Adam

Combinations are possible:

- Mini-batch + Vanilla GD
- Mini-batch + Momentum
- Mini-batch + RMSProp
- Mini-batch + Adam

My suggestion for your deep-learning studies: Keep these two dimensions separate in your mind. Much of the apparent complexity of optimisers disappears once you realise that one decision concerns calculating the gradient, while the other concerns using it.

That is the mystery unfolded.
