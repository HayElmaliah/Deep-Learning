r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 128
    hypers["seq_len"] = 128
    hypers["h_dim"] = 256
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.0005
    hypers["lr_sched_factor"] = 0.07
    hypers["lr_sched_patience"] = 0.7
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.9
    start_seq = "ACT I."
    # start_seq = '''
    # ACT I. SCENE I.
    # A moonlit hill. Enter CHORUS.
    
    # CHORUS:
    # Upon yon hill, where moonlight casts its glow,
    # Beneath the veil of night, a tale shall grow.
    # In fair Verona's streets, where lovers weep,
    # Two houses, both alike in dignity, keep
    # Their ancient grudge, which breaks to new revolt,
    # And fills the stage with tragedy untold.
    
    # ACT I. SCENE II.
    # A bustling street in Verona. Enter SAMPSON and GREGORY, servants to the houses of MONTAGUE and CAPULET.
    
    # SAMPSON:
    # Gregory, by my sword, we shall not bear this affront!
    # Draw your blade, and let our foes confront.
    
    # GREGORY:
    # Hold, Sampson! Let us keep the peace awhile,
    # For brawls and bloodshed only breed more guile.
    
    # SAMPSON:
    # Thou speakest true, but by my maidenhead,
    # I long to strike the Montagues with dread.
    
    # GREGORY:
    # Peace, good Sampson! Here comes Abram, a Montague.
    # Let us provoke him with our biting tongue.
    
    # Enter ABRAHAM, a servant of the Montagues.
    
    # ABRAHAM:
    # What ho! You Capulet dogs! Draw, if you dare!
    # I shall defend my master's honor fair.
    
    # SAMPSON:
    # A Montague, thou sayest? Prepare for strife!
    # This day shall mark the ending of thy life!
    
    # GREGORY:
    # Nay, Sampson, prithee, let us not be rash.
    # A fight like this will bring forth no good clash.
    
    # ABRAHAM:
    # I mock thee, fools! Thy courage is but air.
    # But know this, Capulets: the Montagues won't bear
    # Such insolence. We'll meet thee on the field,
    # And make thy Capulet hearts surely yield.
    
    # Exit ABRAHAM.
    
    # CHORUS:
    # Thus, the stage is set, the feud ablaze,
    # A tragic tale of love and death's embrace.
    # In fair Verona, where our story's told,
    # Let fate and passion now unfold.
    # '''
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text for a few reasons. It helps maintain contextual relationships between close sentences and avoids learning irrelevant patterns. Additionally, it mitigates issues with vanishing or exploding gradients. Training on smaller sequences allows for faster training and better capturing of localized context. Overall, splitting the corpus into sequences improves learning, addresses gradient-related problems, and enhances training efficiency.
"""

part1_q2 = r"""
**Your answer:**

The generated text can show memory longer than the sequence length due to the utilization of hidden states in the model. Similar to human memory mechanisms, the model maintains a hidden state that carries information from previous batches. This allows the model to access context beyond the current sequence length and incorporate it into the generated text. By propagating the hidden states throughout successive batches in the training loop, the model benefits from the accumulated context and demonstrates the ability to exhibit longer-term memory in the generated output. This combination of hidden states and sequential training enables the model to generate coherent and contextually rich text beyond the limitations of the individual sequence length.
"""

part1_q3 = r"""
**Your answer:**

We do not shuffle the order of batches when training in order to preserve the sequential nature and continuity of the text. Language and sentences carry crucial context, and their order plays a significant role in conveying meaning. Shuffling the batches would disrupt this inherent structure and potentially lead to the loss of coherent and meaningful sequences. Additionally, by maintaining the order of batches, we ensure that the hidden states in the model carry forward relevant information from one batch to the next. This continuity allows the model to effectively leverage the context provided by the sequential nature of the text, enabling it to generate coherent and contextually appropriate responses. By keeping the order intact, we facilitate the model's ability to understand and generate text that adheres to the linguistic structure and maintains the desired flow of information.
"""

part1_q4 = r"""
**Your answer:**

4.1.
We lower the temperature for sampling (compared to the default of 1.0) to control the diversity and randomness of the generated text. By decreasing the temperature, we make the probability distribution less uniform and give more weight to characters with higher scores. This allows us to generate text that aligns more closely with the model's confident predictions and reduces the likelihood of sampling less likely characters.

4.2.
When the temperature is very high, the probability distribution becomes more uniform. This means that all characters have a similar probability of being selected, regardless of their scores. As a result, the generated text becomes more random and less meaningful. The high temperature encourages exploration of various possibilities but can lead to less coherent and structured output.

4.3.
When the temperature is very low, the probability distribution becomes highly peaked or one-hot encoded. This means that the next character is predominantly determined by the character with the highest score. The low temperature makes the model more deterministic and focused on the most likely predictions. This can result in repetitive patterns and a lack of variability in the generated text.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 512
    hypers['z_dim'] = 64
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = 0.9, 0.999
    hypers['x_sigma2'] = 0.0005
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


The hyperparameter sigma^2 in VAE determines the variance of the likelihood distribution P(X|Z), where Z represents a sample in the latent space and X is an instance in the instance space. Its role is crucial in controlling the trade-off between reconstruction accuracy and regularization in the VAE framework.

When sigma^2 is set to a low value, it increases the weight of the reconstruction loss in the total loss function. This prioritizes the fidelity of reconstructing the input data from the latent space. Then, the VAE tends to produce samples that closely resemble the training data. The low variance encourages the latent space to be more focused and samples drawn from it are likely to be similar to each other. While this can lead to faithful reconstructions, it may also limit the diversity of generated outputs.

When sigma^2 is set to a high value, it reduces the emphasis on the reconstruction loss and gives more weight to the regularization term, typically measured by the KLD loss. This encourages the latent space to have a broader distribution, allowing for greater exploration and generating more diverse samples. The higher variance allows the model to capture different modes of the data distribution, potentially leading to novel and creative outputs. However, excessively high variance values can result in generated samples that deviate significantly from the training data or lack coherence.
"""

part2_q2 = r"""
**Your answer:**


2.1.
The VAE loss term consists of two components:

- The reconstruction loss measures the similarity between the model's output and the original input data. Its purpose is to ensure that the generated samples closely resemble the training data. By minimizing the reconstruction loss, the VAE aims to reconstruct the input data accurately, encouraging the latent space to capture the essential features of the original data distribution.

- The KL divergence loss quantifies the difference between the distribution of latent vectors and a desired prior distribution in the latent space. Minimizing the KL divergence loss encourages the latent space distribution to resemble the prior distribution. This regularization term helps maintain the latent vectors in a dense and structured space, making them more interpretable and ensuring that samples drawn from this space are meaningful and coherent.

2.2.
The KL divergence loss term influences the shape and characteristics of the latent-space distribution in a VAE. It measures the discrepancy between the actual distribution of latent vectors and the desired prior distribution, typically a standard Gaussian distribution.
Minimizing the KL divergence loss encourages the latent-space distribution to approach the prior distribution, making it more Gaussian-like. This effect helps to regularize and structure the latent space, ensuring that the latent vectors capture meaningful representations of the input data. It promotes smoothness and continuity in the latent space, allowing for meaningful interpolation and exploration between different data points.

2.3.
The benefit of this effect is two-fold:
- By enforcing a latent-space distribution that approximates a standard Gaussian, we can apply sampling techniques to generate new data points. These samples will have coherent and meaningful representations, ensuring high-quality generated outputs that resemble the original data distribution.
- The regularization effect of the KL divergence loss prevents overfitting and encourages the VAE to learn robust and generalizable representations. It helps in disentangling the underlying factors of variation in the data, making the latent space more interpretable and facilitating tasks such as data manipulation, interpolation, and generation."""

part2_q3 = r"""
**Your answer:**

In the formulation of the VAE loss, starting by maximizing the evidence distribution p(x) serves a crucial purpose:
By maximizing the evidence distribution, we aim to learn the parameters that describe the distribution of the original data.
Maximizing the evidence distribution allows us to evaluate how well the VAE can reconstruct instances from the latent space. If we can accurately project instances from the latent space and reconstruct them, it indicates that the VAE has captured the essential features and patterns of the data. This means that when we sample new instances and decode them, they are likely to resemble the original data.
"""

part2_q4 = r"""
**Your answer:**

In the VAE encoder, we model the log of the latent-space variance instead of directly modeling the variance itself. This choice is motivated by the presence of very small positive values in the variance, which can pose challenges for the model to learn effectively. By taking the logarithm, the values are scaled to a larger range, allowing the model to capture smaller and more accurate results. Additionally, the log transform provides numerical stability, as it maps the small positive values to a wider range of values, enabling smoother optimization. Hence, by modeling the log variance, we mitigate numerical instability and facilitate more accurate and stable learning in the VAE encoder.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers["embed_dim"] = 128
    hypers["num_heads"] = 8
    hypers["num_layers"] = 4
    hypers["hidden_dim"] = 128
    hypers["window_size"] = 128
    hypers["droupout"] = 0.25
    hypers["lr"] = 0.0005
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**


Stacking encoder layers that employ sliding-window attention leads to a broader context in the final layer by progressively capturing and integrating information from an expanding contextual window. Similar to how CNNs stack layers to capture larger spatial patterns, stacking encoder layers with sliding-window attention allows the model to incorporate a broader range of dependencies in the input sequence. Each encoder layer attends to a fixed-sized window of neighboring positions using sliding-window attention. This attention mechanism focuses on relevant parts within the window, capturing local interactions and dependencies. By stacking multiple encoder layers, the model sequentially processes the output of each layer, which already includes information from a wider context. This enables the subsequent layers to capture longer-range dependencies and incorporate a more comprehensive understanding of the input sequence. As a result, the final layer of the stacked encoder layers encompasses a broader context, as each layer has successively integrated information from a larger contextual window.
"""

part3_q2 = r"""
**Your answer:**


The Multi-Scale Attention Fusion approach involves a multi-step process to capture both local and global context. Firstly, a set of different window sizes or scales is defined, each representing a specific contextual range. Attention patterns are then computed independently for each scale using sliding-window attention, extracting contextual information within the corresponding window size. This step ensures that both local dependencies and broader interactions are considered.

Next, scale-specific weights are assigned to determine the relative importance of each scale. These weights can be dynamically determined based on their relevance to the task or predefined using specific criteria. The purpose of these weights is to control the contribution of each scale during the attention fusion process, allowing for a balanced integration of information from different contextual ranges.

The attention fusion step combines the attention patterns from different scales using the assigned weights. This fusion process aggregates the attended information from various scales, resulting in a final attention representation that captures both local and global context. By fusing attention patterns from multiple scales, the model can incorporate information from different contextual ranges, enabling a more comprehensive understanding of the input sequence.

An additional refinement step can be performed (optionally) to recalibrate or adjust the attention distribution based on the fused attention pattern. This step fine-tunes the attention representation, ensuring that it aligns with the specific requirements of the task at hand. It allows for further enhancement of the integration of local and global context, leading to more effective information utilization.

This variation maintains a similar computational complexity to sliding-window attention since attention is computed independently at each scale, with the fusion step being an additional computational cost.
"""


part4_q1 = r"""
**Your answer:**

We fine-tuned a pre-trained Distil-BERT model for sentiment analysis using two methods: Method 1 involved freezing most layers and training only the final ones, while Method 2 retrained all parameters. Test accuracies were more than 80% for both Method 1 and Method 2 (when model 2 accuracy was higher then that of model 1). Comparing these results to the trained-from-scratch encoder in Part 3, which achieved a test accuracy of 67.9%, we can observe that the fine-tuned BERT models outperformed the encoder trained from scratch.

There are several factors that can contribute to the improved performance of the fine-tuned BERT models. Firstly, BERT is a powerful language model pre-trained on a large corpus of data, enabling it to capture rich contextual representations. By fine-tuning BERT on the specific sentiment analysis task, we leverage its pre-trained knowledge and allow it to adapt to the nuances of sentiment classification. Additionally, fine-tuning methods involve updating the weights of the pre-trained model based on task-specific data. In method 1, we froze the majority of the model's layers and only trained the last few linear layers. This approach is suitable when limited labeled data is available or when the pre-trained model is already knowledgeable in a similar domain. Method 2 involved retraining all the parameters of the model, allowing it to learn task-specific features from scratch or adapt its representations more flexibly.

The superior performance of the fine-tuned BERT models in this specific task does not necessarily guarantee the same outcome in any downstream task. The effectiveness of pre-training and fine-tuning depends on various factors such as the size and diversity of the pre-training data, the similarity of the pre-training task to the downstream task, and the availability of labeled data for fine-tuning. Different tasks may require different strategies and approaches, and the performance of pre-trained models may vary across tasks.
"""

part4_q2 = r"""
**Your answer:**

When fine-tuning a pre-trained model, it's usually better to freeze the internal layers closer to the input and only update the last layers, like the classification head. This is because the deeper layers capture specific patterns for the task, while the earlier layers capture more general information.

If we freeze internal layers like the multi-headed attention blocks, the model can struggle to learn specific patterns and task-related details, resulting in poorer performance. Freezing these layers also disrupts the fine-tuning process by preventing them from adapting to the task-specific data. However, the impact of unfreezing internal layers can vary. If the pre-training and fine-tuning tasks are similar and we have enough data, unfreezing some layers may not have a big effect. But if the tasks are different or we have limited data, unfreezing internal layers can lead to overfitting or difficulties in fine-tuning due to increased complexity.

It's generally easier and more effective to freeze the internal layers and update only the last layers during fine-tuning. This allows the model to leverage its pre-trained knowledge while adapting to the specific task.
"""


# ==============
