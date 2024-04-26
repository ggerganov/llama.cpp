# Perplexity

The `perplexity` example can be used to calculate the so-called perplexity value of a language model over a given text corpus.
Perplexity measures how well the model can predict the next token with lower values being better.
Note that perplexity is **not** directly comparable between models, especially if they use different tokenizers.
Also note that finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases.

Within llama.cpp the perplexity of base models is used primarily to judge the quality loss from e.g. quantized models vs. FP16.
The convention among contributors is to use the Wikitext-2 test set for testing unless noted otherwise (can be obtained with `scripts/get-wikitext-2.sh`).

By default only the mean perplexity value and the corresponding uncertainty is calculated.
The uncertainty is determined empirically by assuming a Gaussian distribution of the "correct" logits per and then applying error propagation.

More statistics can be obtained by recording the logits from the FP16 version of a model.
To do this, supply `perplexity` with `--kl-divergence-base path/to/logit/binary/file.kld`.
The program will then record all logits and save them to the provided path in binary format.
**The logit file will be very large, 11 GiB for LLaMA 2 or 37 GiB for LLaMA 3 when using the Wikitext-2 test set.**
Once you have the file, supply `perplexity` with the quantized model, the logits file via `--kl-divergence-base`,
and finally the `--kl-divergence` argument to indicate that the program should calculate the so-called Kullback-Leibler divergence.
This is a measure of how similar the FP16 and the quantized logit distributions are with a value of 0 indicating that the distribution are the same.
The uncertainty on the mean KL divergence is calculated by assuming the KL divergence per token follows a Gaussian distribution.

In addition to the KL divergence the following statistics are calculated with `--kl-divergence`:

* Ratio of mean FP16 PPL and quantized PPL. Uncertainty is estimated on logits, then propagated. The logarithm of this metric is also calculated and printed, it is 0 if the logit distributions are the same.
* Difference of mean FP16 PPL and quantized PPL. Uncertainty is estimated on logits, then propagated.
* Mean change in "correct" token probability. Positive values mean the model gets better at prediction, negative values mean it gets worse.
* Pearson correlation coefficient of the "correct" token probabilites between models.
* Percentiles of change in "correct" token probability. Positive values mean the model gets better at prediction, negative values mean it gets worse. Can be used to judge noise vs. quality loss from quantization. If the percentiles are symmetric then the quantization is essentially just adding noise. If the negative values are significantly larger than the positive values then this indicates that the model is actually becoming worse from the quantization.
* The root mean square of the change in token probabilities. If you were to assume that the quantization simply causes Gaussian noise on the token probabilities then this would be the standard deviation of said noise. The uncertainty on the value is calculated that the change in token probabilities follows a Gaussian distribution. Related discussion: https://github.com/ggerganov/llama.cpp/discussions/2875 .
* Same top p: Percentage of how often the token was assigned the highest probabilites by both models. The uncertainty is calculated from the Gaussian approximation of the binomial distribution.

## Sample results

| Metric          |          L2 7b q2_K |           L3 8b q2_K |        L2 7b q4_K_M |        L3 8b q4_K_M |          L2 7b q6_K |          L3 8b q6_K |          L2 7b q8_0 |          L3 8b q8_0 |
|-----------------|---------------------|----------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Mean PPL        | 5.794552 ± 0.032298 | 10.641563 ± 0.071555 | 5.877078 ± 0.032781 | 6.956203 ± 0.044123 | 5.808494 ± 0.032425 | 6.796525 ± 0.043079 | 5.798542 ± 0.032366 | 6.764558 ± 0.042733 |
| Mean PPL ratio  | 1.107955 ± 0.001427 |  1.574354 ± 0.004748 | 1.014242 ± 0.000432 | 1.029127 ± 0.000761 | 1.002406 ± 0.000191 | 1.005504 ± 0.000314 | 1.000689 ± 0.000107 | 1.000775 ± 0.000174 |
| Mean ΔPPL       | 0.625552 ± 0.008725 |  3.882242 ± 0.038457 | 0.082526 ± 0.002530 | 0.196882 ± 0.005284 | 0.013941 ± 0.001110 | 0.037204 ± 0.002156 | 0.003990 ± 0.000624 | 0.005237 ± 0.001179 |
| PPL correlation |              97.36% |               89.48% |              99.71% |              99.32% |              99.94% |              99.88% |              99.98% |              99.96% |
| Mean KLD        | 0.108903 ± 0.000645 |  0.456326 ± 0.001851 | 0.012686 ± 0.000079 | 0.031046 ± 0.000249 | 0.002098 ± 0.000014 | 0.004726 ± 0.000045 | 0.000369 ± 0.000007 | 0.000753 ± 0.000005 |
| Mean Δp         |    -2.710 ± 0.023 % |     -9.059 ± 0.051 % |    -0.416 ± 0.008 % |    -0.602 ± 0.014 % |    -0.035 ± 0.003 % |    -0.018 ± 0.006 % |    -0.005 ± 0.002 % |    -0.022 ± 0.002 % |
| Maximum Δp      |             85.136% |              96.663% |             45.209% |             95.551% |             23.593% |             68.137% |             43.925% |             20.194% |
| 99.9% Δp        |             37.184% |              54.277% |             17.461% |             27.125% |              7.798% |             12.584% |              3.387% |              5.013% |
| 99.0% Δp        |             18.131% |              26.536% |              7.798% |             11.451% |              3.838% |              5.524% |              1.867% |              2.454% |
| Median Δp       |             -0.391% |              -2.430% |             -0.026% |             -0.025% |             -0.001% |             -0.000% |             -0.000% |             -0.000% |
| 1.0% Δp         |            -39.762% |             -86.286% |            -11.433% |            -19.147% |             -4.222% |             -5.949% |             -1.862% |             -2.563% |
| 0.1% Δp         |            -79.002% |             -98.787% |            -26.433% |            -55.577% |             -9.091% |            -16.975% |             -3.252% |             -5.260% |
| Minimum Δp      |            -99.915% |             -99.963% |            -83.383% |            -98.976% |            -43.142% |            -83.219% |             -9.343% |            -17.047% |
| RMS Δp          |     9.762 ± 0.053 % |     21.393 ± 0.078 % |     3.252 ± 0.024 % |     5.429 ± 0.051 % |     1.339 ± 0.010 % |     2.096 ± 0.029 % |     0.618 ± 0.011 % |     0.867 ± 0.007 % |
| Same top p      |    85.584 ± 0.086 % |     70.419 ± 0.120 % |    94.665 ± 0.055 % |    92.162 ± 0.071 % |    97.520 ± 0.038 % |    96.586 ± 0.048 % |    98.846 ± 0.026 % |    98.467 ± 0.032 % |

<details>
<summary>Old numbers</summary>

## Llama 2 70B Scorechart

| Quantization | Model size (GiB) | Perplexity | Delta to fp16 |
|--------------|------------------|------------|---------------|
| Q4_0         | 36.20            | 3.5550     | 3.61%         |
| Q4_1         | 40.20            | 3.5125     | 2.37%         |
| Q5_0         | 44.20            | 3.4744     | 1.26%         |
| Q2_K         | 27.27            | 3.7339     | 8.82%         |
| Q3_K_S       | 27.86            | 3.7019     | 7.89%         |
| Q3_K_M       | 30.83            | 3.5932     | 4.72%         |
| Q3_K_L       | 33.67            | 3.5617     | 3.80%         |
| Q4_K_S       | 36.39            | 3.4852     | 1.57%         |
| Q4_K_M       | 38.54            | 3.4725     | 1.20%         |
| Q5_K_S       | 44.20            | 3.4483     | 0.50%         |
| Q5_K_M       | 45.41            | 3.4451     | 0.40%         |
| Q6_K         | 52.70            | 3.4367     | 0.16%         |
| fp16         | 128.5            | 3.4313     | -             |

<details>
