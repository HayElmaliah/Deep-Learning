r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""

1.A:

With the following tensors as described in the question:
$$ {Y}\in{R}^{64\times 512} $$
$$ {X}\in{R}^{64\times 1024} $$
We get by calculus that:
$$
\frac{\partial Y}{\partial{X}}
$$
is of size:
$$
64 \times 512 \times 64 \times 1024
$$


1.B:

The Jacobian is sparse and only the values $\frac{\partial Y_{i,l}}{\partial{X}_{i,k}}$ are non-zero.
That is since each output element depends only in the corresponding input element.

1.C:

We don't need to materialize the above Jacobian in order to calculate the downstream gratdient w.r.t. to the input ($\delta{X}$).
Since we have the gradient of the output with respect to the loss, denoted as $\delta{Y}=\frac{\partial L}{\partial{Y}}$, using the chain rule we get:
$$\delta{X}=\frac{\partial L}{\partial{X}} = \frac{\partial L}{\partial{Y}}\cdot W^{T}$$

2.A:

With the following tensors as described in the question:
$$ {Y}\in{R}^{64\times 512} $$
$$ {W}\in{R}^{512\times 1024} $$
We get by calculus that:
$$
\frac{\partial Y}{\partial{X}}
$$
is of size:
$$
64 \times 512 \times 512 \times 1024
$$

2.B:

Same as above, we get that Jacobian is sparse and only the values $\frac{\partial Y_{i,l}}{\partial{X}_{i,k}}$ are non-zero.
That is since each $Y_{i}$ is a linear-combination of the $i_{th}$ row of $W$.


2.C:

Same as above - we don't need to materialize the above Jacobian in order to calculate the downstream gratdient w.r.t. to the input. We again use the chain rule and get:
$$\delta{W}=\frac{\partial L}{\partial{W}} = \frac{\partial L}{\partial{Y}}\cdot W^{T}$$ 


"""

part1_q2 = r"""
**Your answer:**

Back-propagation is not required in order to train neural networks with gradient-based optimization. Alternative methods, such as derivative-free optimization techniques like the Nelder-Mead method, exist. However, these methods often lack the efficiency and accuracy provided by back-propagation. While it is technically possible to calculate the entire derivative without using the chain rule, this approach is generally impractical and inefficient. Back-propagation, on the other hand, offers significant advantages in terms of computational efficiency and accuracy. Back-propagation is a specific algorithm that efficiently calculates gradients by leveraging the chain rule, computational graphs, and automatic differentiation. It allows for the propagation of errors through the network and enables the adjustment of network weights based on these gradients.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.05
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr_vanilla = 0.02
    lr_momentum = 0.004
    lr_rmsprop = 0.0002
    reg = 0.0015
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.15
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1.1.

The graphs comparing the no-dropout and dropout configurations align with our expectations. Without dropout, the training accuracy remains more stable across epochs, showing fewer fluctuations compared to the dropout configurations. This is because all neurons are active during training without dropout, leading to stronger and more focused connections within the network. As a result, the model tends to perform well on the training data, resulting in higher training accuracy.
On the other hand, the introduction of dropout introduces more spikes in the accuracy curves between epochs. Dropout randomly "drops out" a fraction of neurons during each training iteration, encouraging the network to distribute the representation across multiple sets of neurons. This promotes better generalization and reduces overfitting. As a consequence, the model may exhibit lower training accuracy compared to the no-dropout case.
In terms of test accuracy, the results show a trade-off between the dropout configurations and the no-dropout case. Without dropout, the model achieves higher test accuracy since it has learned strong and specific connections tailored to the training data. However, this can lead to overfitting, where the model fails to generalize well to unseen data.

1.2.

As the dropout rate increases, we observe a decrease in test accuracy but an improvement in generalization. For example, comparing a low dropout rate (e.g., 0.4) to no dropout, the test accuracy may slightly decrease, but it helps in reducing overfitting and improving performance on unseen data. Dropout prevents the model from relying too heavily on specific neurons, encouraging the network to learn more robust and generalizable features.
However, it's important to note that extremely high dropout rates, such as 0.8, can harm both training and testing accuracy. With a dropout rate of 0.8, a significant portion of neurons is disabled during training, severely limiting the model's capacity to learn meaningful patterns from the data. As a result, the model's performance is likely to suffer, yielding poor results for both training and testing.

"""

part2_q2 = r"""
**Your answer:**

Yes, it is possible. It might ocuur in some scenarios and that is due to the behavior of the Cross-Entropy loss function.

The Cross-Entropy loss function peneltizes for predicted labels $\hat{y}$ that are far (in terms of distance) from the true labels $y$. But the accuracy only cares for the predicted labels to be equal to the true labels.

For example, we look at some $y_1 = \hat{y}_1$  and $y_2 \ne \hat{y}_2, y_3 \ne \hat{y}_3$ that are very close to each other. Then, in that epoch, $\hat{y}_2, \hat{y}_3$ increase **a bit**, such that $y_2 = \hat{y}_2, y_3 = \hat{y}_3$ and $y_1$ decreases **a lot**.

In that case, all the predicted label have not changed except for two that are now equal to their true labels and one that now is not. Thus the test accuracy increases.

On the other hand, we get in total greater distances between the predicted and the true labels and thus the loss also increses.


"""

part2_q3 = r"""
**Your answer:**

3.1:
> **Gradient descent** is an algorithm used for optimization. GD used to minimize the loss function, by updating iteratively the hyper parameters.

> **Backpropagation** is an algorithm used for efficient calculation (using the chain rule) of the derivatives of the loss function w.r.t the parameters.

3.2:
> GD and SGD are both optimization algorithms used in training machine learning models. They differ in the way they updating the model's parameters:

> In GD, in each epoch, the algorithm cosider the whole dataset $X$ to decide and perform the step. on the contrary, in GSD, in each epoch the algorithm samples a subset of the dataset $X$, which its size is $BatchSize$, and only take that subset into considration whlie deciding the step.

> The GD is more robust and less sensitive to noicy data, compare to GSD, Because it uses the entire data. Furthermore, GSD might not get to the minima (the solution may fluctuate around the optimal point), Because it is effected by individual samples. Nevertheless, The GSD take smaller steps and thus converges faster than GD.

3.3:
> Following are some reasons for that:
> * SGD uses only a part of the dataset, allowing efficient and possible optimization for big datasets, which is sometimes not even possible for them to run GD.
> * SGD uses different samples in each epoch. That allows it sometimes to converge better, since it can ignore some noisy data wich might lead to bad steps.
> * SGD converges faster, since it takes smaller steps, as we explained above.

"""

part2_q4 = r"""
**Your answer:**

4.1.

To reduce the memory complexity for computing the gradient using forward mode Automatic Differentiation (AD) while maintaining O(n) computation cost, we can employ a technique called checkpointing. The idea is to only store the essential values needed for gradient computation instead of storing all intermediate results. In the first approach, the algorithm initializes two variables, currentGradient and currentResult, and iterates through the computational graph in a forward pass. At each step, it computes the derivative of the current function and updates the gradient and result accordingly. By only storing the current gradient and result, the memory complexity is reduced to O(1). However, if the intermediate results are not already given, the memory complexity becomes O(n) as we need to store all the intermediate results.

4.2.

Similarly, for backward mode AD, we can use checkpointing to reduce memory complexity while maintaining O(n) computation cost. The second approach initializes two holders, backwardGradient and backwardResult, and performs a forward pass through the computational graph, saving the intermediate function results. Then, in the backward pass, it iterates from the end to the beginning, computing the gradients based on the saved function results. By only storing the necessary function results, the memory complexity is reduced to O(1).

4.3.

These techniques leverage the concept of checkpointing to minimize memory usage during gradient computation. However, it's important to note that these techniques assume a sequential execution of functions and may not be optimal for computational graphs with parallel executions. The memory complexity reduction to O(1) holds under the assumption of a sequential execution.
These memory optimization techniques can be generalized for arbitrary computational graphs by employing checkpointing and breaking down the graph into subgraphs. By computing the gradient of each subgraph separately and combining them, we can reduce the memory complexity of the overall gradient computation to O(1). As long as we store only the essential values needed for gradient computation, we can handle large and complex computational graphs efficiently.

4.4.

In the context of deep architectures such as VGGs and ResNets, these memory optimization techniques offer significant benefits. These architectures often have a large number of parameters and layers, resulting in high memory requirements. By reducing the memory complexity of gradient computation to O(1), we can alleviate the memory burden during training. This advantage becomes crucial, as it enables efficient training on hardware with limited memory resources. Additionally, the reduced memory complexity allows for faster and more scalable training, improving the overall training time of deep architectures.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 512
    activation = "relu"
    out_activation = "relu"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.5
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

Our model has low optimization error. This conclusion is drawn from observing that the loss graph decreases and the accuracy graph increases for the validation set as the model learns. The decreasing loss indicates that the model is effectively optimizing its parameters to minimize the discrepancy between predicted and actual values.

Our model has low generalization error. This conclusion is based on several factors. Firstly, the validation graph shows similar trends to the train graph, suggesting that the model performs well on unseen data. Additionally, the validation accuracy of approximately 90% indicates that the model can successfully classify unseen samples, demonstrating good generalization ability.

Our model has low approximation error. Although there are some indications of approximation error in the decision plots, it is not considered high. The decision boundary is close to optimal, and the validation accuracy of around 90% further supports this assessment. If the model had underfitted, we would expect a significantly lower validation accuracy.

"""

part3_q2 = r"""
**Your answer:**

We would expect the False Negative Rate (FNR) to be higher than the False Positive Rate (FPR) for the validation dataset. The reasoning behind this expectation is that the training dataset's plot reveals a higher concentration of positive samples surrounded by negative samples. As a result, the learned decision boundary is likely to classify more positive samples as negative. This tendency carries over to the validation dataset, resulting in a higher FNR.

"""

part3_q3 = r"""
**Your answer:**

In the scenario where a person with the disease will develop non-lethal symptoms that confirm the diagnosis, the focus would be on lowering the False Positive Rate (FPR) to minimize the costs associated with unnecessary follow-up tests, while accepting a slightly higher False Negative Rate (FNR) given the low-risk nature of the disease and the eventual appearance of detectable symptoms. Therefore, the preferred point on the ROC curve would have a low 1-FNR and a corresponding low False Positive Rate (FPR), such as the point (FPR, TPR) = (0, 0).

In the scenario where a person with the disease shows no clear symptoms and faces a high probability of death without early diagnosis, the focus shifts to saving lives. Here, it becomes essential to prioritize a low FNR to identify those individuals at risk and provide timely intervention. Even if it incurs additional costs (higher FPR), the primary objective is to minimize the loss of life. Consequently, the preferred point on the ROC curve would have a low FNR, such as the point (FPR, TPR) = (0, 1).

"""


part3_q4 = r"""
**Your answer:**

4.1.

For the case of depth=1, it was observed that as the width increased, the validation and test accuracies decreased. This trend was also observed for depth=2. However, for depth=4, the val and test accuracy initially decreased with increasing width, but for the largest width value (32), the accuracy improved and almost reached the level observed with width=2 (87% compared to the initial accuracy of 89%). Overall, the best performance was achieved when the depth was set to 2 and the width was set to 2, resulting in an (test) accuracy of 91.

Regarding the decision boundaries, it was observed that for any fixed depth, increasing the width led to more flexible and complex decision boundaries. This suggests that wider models have the ability to capture more intricate patterns and relationships within the data, enabling them to classify samples with higher complexity.

4.2.

Upon analyzing the results for fixed width and varying depths, several patterns emerge. For smaller widths, such as 2 and 8, increasing the depth initially improves the accuracy of the model. However, there is a threshold beyond which increasing the depth no longer benefits the model, and the accuracy starts to decline. This decline is even more pronounced, with the accuracy dropping below the initial values. For instance, with width=2, the test accuracy starts at 90%, increases to 91.5% for depth=2, but then decreases to 89.5% for depth=8. A similar trend is observed for width=8, starting at 87%, increasing to 91%, and then dropping to 84.3%.
Interestingly, for the largest width value of 32, increasing the depth leads to a consistent improvement in accuracy. The test accuracy increases from 84.1% for depth=1 to 86.8% for depth=2, and further to 87.8% for depth=8. This suggests that deeper networks are more effective in capturing the complexity of the data when the width is larger.

In terms of decision boundaries, the effect of increasing the depth is more pronounced for smaller width values (2 and 8). In these cases, increasing the depth results in more flexible and complex decision boundaries. On the other hand, for width=32, there is less noticeable difference in the decision boundaries as the depth increases, indicating that the initial depth may already capture the complexity of the data fairly well.

These observations align with the understanding that increasing the depth of an MLP can enhance its representational capacity and ability to learn intricate patterns in the data. However, there is a trade-off, as excessively deep networks may suffer from issues like vanishing gradients or overfitting. Therefore, striking the right balance between depth and width is crucial to achieve optimal performance and decision boundary complexity.

4.3.

When comparing the results for configurations with the same number of total parameters, namely depth=1 and width=32 (referred to as Configuration A), and depth=4 and width=8 (referred to as Configuration B), several observations can be made.
Firstly, in Configuration A, we observe a higher threshold value of 0.32 compared to 0.28 in Configuration B. The threshold value indicates the decision boundary of the model, and a higher threshold suggests a more conservative classification approach.
Secondly, when considering the validation and test accuracies, Configuration B performs slightly better than Configuration A. The validation accuracy for Configuration B is 82.7%, while for Configuration A it is 80.6%. Similarly, the test accuracy for Configuration B is 84.3%, whereas for Configuration A it is 84.1%. These results indicate that Configuration B achieves slightly higher accuracy on unseen data.
In terms of decision boundaries, both configurations exhibit similar patterns. The decision boundaries in both cases appear to be comparable, suggesting that the models have similar capabilities in separating and classifying different data points.
Overall, when comparing Configuration A (depth=1, width=32) and Configuration B (depth=4, width=8), Configuration B shows slightly better performance in terms of validation and test accuracies while maintaining similar decision boundary patterns. This implies that increasing the depth to 4 and reducing the width to 8 leads to improved model performance in this scenario.

4.4.

The effect of threshold selection on the validation set and its impact on the test set can be complex and dependent on various factors. In our case, we can see that in 5 out of 9 instances, the test accuracy is higher than the validation accuracy after applying the chosen threshold. This suggests that, to some extent, the threshold selection improved the results on the test set.
However, it is important to consider the general principles and potential limitations associated with threshold selection. In some cases, the optimal threshold determined based on the validation set does not necessarily lead to improved results on the test set. The reason for this lies in the sensitivity of the optimal threshold to the specific dataset used for validation.
Although the validation set is intended to represent an independent distribution and provide an estimate of model performance, it does not guarantee identical results on the test set. Differences in the distributions and characteristics of the validation and test sets can lead to variations in the optimal threshold. Therefore, selecting the optimal threshold solely based on the validation set may not always yield the best performance on the test set.


"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.05
    weight_decay = 0.005
    momentum = 0.05
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1.1.

Number of parameters in the regular block:
The first convolutional layer has a kernel size of 3x3 and 256 input channels. With an additional bias term, it gives us (3x3x256 + 1) parameters for each of the 256 output channels. Therefore, the total number of parameters for the first convolutional layer is (3x3x256 + 1) x 256. The second convolutional layer also has a kernel size of 3x3 and 256 input channels. Similar to the first layer, it yields (3x3x256 + 1) parameters for each of the 256 output channels. Thus, the total number of parameters for the second convolutional layer is (3x3x256 + 1) x 256. Since there are two convolutional layers in the regular block, we get (3x3x256 + 1) x 256 x 2 = 1,180,160.

Number of parameters in the bottleneck block:
The first convolutional layer is a 1x1 convolution that reduces the 256 input channels to 64 output channels. Including the bias term, this gives us (1x1x256 + 1) parameters. The second convolutional layer has a kernel size of 3x3 and operates on the 64 input channels. With an additional bias term, it yields (3x3x64 + 1) parameters for each of the 64 output channels. The third and final convolutional layer is another 1x1 convolution that expands the 64 input channels back to 256 output channels. Including the bias term, this gives us (1x1x64 + 1) parameters. Total number of parameters in the bottleneck block = (1x1x256 + 1) x 64 + (3x3x64 + 1) x 64 + (1x1x64 + 1) x 256 = 70,016.

In terms of the number of parameters, we can observe that the bottleneck block has significantly fewer parameters (70,016) compared to the regular block (1,180,160). This reduction in the number of parameters can lead to faster training and reduced computational requirements.

1.2.

The number of floating point operations is directly proportional to the size of the kernel and the number of parameters in each layer. Thus, the regular network requires a significantly larger number of calculations compared to the bottleneck network. The number of floating point calculations is given by: number_of_params∗(image_size−(kernel_size−stride))^2.

1.3.

The regular block, which consists of two 3x3 convolutions, excels in combining input within feature maps due to its ability to capture spatial relationships and patterns within each layer. With two convolutions, it has a wider receptive field, allowing it to consider more features within the feature map. On the other hand, the bottleneck block utilizes a single 3x3 convolution, limiting its ability to capture fine-grained details within the feature map.

In terms of combining input across feature maps, both the regular and bottleneck blocks demonstrate similar capabilities. Both blocks maintain the same number of input and output channels for each convolution layer, enabling them to combine information across the feature maps effectively. The bottleneck block, however, benefits from transitioning between a higher number of channels (e.g., from 256 to 64), facilitating the integration of high-level and low-level information across feature maps.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1.1.

Analyzing the graphs, it becomes evident that increasing the network depth resulted in a decline in accuracy, best given by shallowest depths 2 and 4, and the depth of 4 yielded the best results for both k=32 and k=64.
a possible explanation could be overfitting: Deeper networks generally have a higher capacity to learn complex representations, but they are also more susceptible to overfitting. The depth of 4 may have been effective in preventing overfitting by striking a balance between capturing relevant features and avoiding excessive model complexity. By not going deeper, the network might have avoided overemphasizing noise or irrelevant patterns, leading to better generalization performance on your dataset.

1.2.

In our experiment, we encountered instances where the network became non-trainable for values of L=8 and L=16. This issue can be attributed to the problem of vanishing gradients, observed when the number of layers exceeds a certain threshold (in our case, above 4). The presence of vanishing gradients causes the gradients to diminish significantly, eventually reaching zero. This phenomenon hinders the model's ability to learn and make updates to its parameters.
To partially address the vanishing gradients problem, two potential solutions can be considered.
Firstly, the utilization of batch normalization can help alleviate this issue. By normalizing the input to each layer, ensuring zero mean and unit variance, the gradients are allowed to flow more smoothly throughout the network without vanishing.
Secondly, incorporating skip connections, inspired by the ResNet architecture, can also mitigate the vanishing gradients problem. By establishing direct connections that bypass upper layers and enable gradients to flow directly to lower layers, the network can maintain a more stable gradient flow during training. This approach promotes better information propagation and enables the model to learn effectively, even with larger values of L.

"""

part5_q2 = r"""
**Your answer:**

Upon analyzing the provided graphs, several observations can be made. First, for a fixed value of L, larger values of K tend to yield higher accuracy in both training and testing. Additionally, in the comparison between L=4 and L=2, L=4 consistently produces superior results.
Similar to the findings in experiment 1.1, we observe that for L=8 (where L>4), the network becomes non-trainable, resulting in extremely low accuracy across all varying values of K. This aligns with our previous understanding.
In comparison to the results from experiment 1.1, where the best performance was achieved with L=4 and K=32, the findings from experiment 1.2 demonstrate even better outcomes. Specifically, we achieve test accuracies surpassing 70% for L=4 and larger values of K such as K=128 and K=256.

"""

part5_q3 = r"""
**Your answer:**

Upon analyzing the graphs, it is evident that the model's performance varies with different values of L (depth). Interestingly, for this experiment, L=3, which corresponds to the second lowest depth, yields the highest test accuracy. This suggests that a moderate depth is optimal for achieving better results in terms of accuracy.

Furthermore, a notable observation is that the accuracy drastically drops for L=4. In previous experiments 1.1 and 1.2, such a significant drop in accuracy occurred only for larger values of L, such as L=8. This indicates the presence of the vanishing gradients phenomenon, where the gradients diminish exponentially as they propagate through the deeper layers of the network.

"""

part5_q4 = r"""
**Your answer:**

Upon analyzing the provided graphs, it becomes evident that shallower depths yield better outcomes in terms of test accuracies. This observation aligns with our earlier findings in section 1.1 and nearly those of section 1.3 where the second lowest depth was ideal in term of accuracy.
When considering the ResNet architecture with a fixed value of K=32, we no longer observe the extremely low accuracy values associated with the vanishing gradients problem, as observed in section 1.1 for L>4 and in section 1.3 for L=4. This improvement is noticeable for L=8 and L=16, suggesting that the vanishing gradients issue has been mitigated to some extent. However, it is worth noting that for L=32, the phenomenon of vanishing gradients still persists.
Interestingly, for larger values of K (>32, such as K=64, 128, 512), and across depths L=2, 4, and 8, we successfully mitigate the vanishing gradients problem, as previously suggested.
Consequently, L=2 yields satisfactory results in terms of accuracy.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1.1.

The performance of YOLO5 on the given images was not satisfactory. In the first image with dolphins, the model struggled to accurately detect the objects and misclassified them as persons or surfboards. It successfully identified some bounding boxes for the dolphins, but the presence of occlusion and the lack of distinguishable features, along with the dolphins being in an unnatural environment, posed challenges for accurate detection.
Similarly, in the second image containing cats and dogs, the model faced difficulties due to the close proximity of the animals. It failed to create separate bounding boxes for each animal and mistakenly labeled them incorrectly. This could be attributed to the similarity between certain dog breeds and cats, leading to model bias, as well as occlusion caused by the overlapping animals.

1.2.

Several factors may have contributed to the poor performance. One potential reason is model bias, where the model's training data might have contained a higher representation of certain classes (e.g., persons on surfboards) compared to others (e.g., flying dolphins). Occlusion, especially in the second image where the cat was partially hidden by the dogs, also hindered accurate detection. Additionally, lighting conditions and the absence of specific classes, such as dolphins, in the trained model could have affected the results.
To address these issues, several suggestions can be considered. Firstly, training the model on a dataset with increased variability per class, including various poses and environmental conditions, would help improve its generalization capabilities. Adjusting the size of bounding boxes could aid in better distinguishing closely positioned objects. Modifying the number of bounding boxes per grid cell could enable the model to locate multiple objects within the same area. Finally, fine-tuning the model using a dataset that encompasses a wider range of classes would provide more comprehensive object recognition capabilities.

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

The model's object detection performance was not accurate in the given images due to certain challenges deliberately introduced.
In the first image of the lion in the forest, the model incorrectly identified the lion as a bear. This misclassification can be attributed to the lion's partial body figure being hidden among the trees, leaving only its face visible. This limited view likely confused the model, leading it to misinterpret the object.
In the second image featuring a keyboard and a hand grenade, the model mistakenly detected the grenade as a computer mouse, albeit with a low probability. This can be attributed to model bias, as it expects to encounter a computer mouse in proximity to a keyboard. This bias influenced the model's interpretation and resulted in an inaccurate detection.
In the third photo depicting a man sitting on a bike, the model detected his leg as a separate body figure, incorrectly associating the helmet with it as the head. Consequently, the model mistakenly identified two people in addition to the bike. This misinterpretation occurred due to the distorted figure of the person and the positioning of the helmet, which created a deceptive appearance that misled the model's detection process.

These intentional challenges were designed to highlight the model's limitations and showcase scenarios where it may struggle to accurately detect objects.

"""

part6_bonus = r"""
**Your answer:**

In an attempt to improve the model performance on poorly recognized images, several manipulations were made to the pictures in the exercise. Here are the modifications that were applied:

Lion in the forest:

Zoomed in to reduce the presence of trees, aiming to match the lion's typical habitat.
Cleared the leaves from the lion's face using Photoshop to enhance its visibility and distinguish it from a bear.
However, despite these changes, the model still misclassified the lion as a bear, albeit with a lower probability.

Keyboard and hand-grenade:

Zoomed in to exclude the cables in the image and avoid displaying the entire keyboard, which could confuse the model.
Drew a white line to create a visual separation between the keyboard and the hand-grenade.
Nevertheless, the modifications did not significantly improve the results. The keyboard was still recognized as a keyboard, while the hand-grenade was detected as a vase. It's possible that the model was not trained on images featuring "dangerous" or "illegal" objects, leading to such misclassifications to avoid potential issues.

Man on bike:

Removed the right part of the image containing the helmet to prevent the model from detecting it as another person.
This alteration successfully resulted in the detection of one person and one bike, compared to the previous identification of two people and one bike.

The modifications attempted to enhance the model's recognition by manipulating the images to align them more closely with the expected objects. However, in some cases, the changes did not produce the desired outcomes, suggesting limitations in the model's training or the complexity of distinguishing certain objects.

It's important to note that improving model performance may require additional steps, such as retraining the model with specialized datasets that include specific objects like hand-grenades or further fine-tuning to adapt to unique detection requirements.


"""