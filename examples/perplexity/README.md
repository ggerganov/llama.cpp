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

## LLaMA 3 8b Scoreboard

Results are sorted by Kullback-Leibler divergence relative to FP16.
The "WT 2.7m" importance matrix was created using 2.7 million Wikitext tokens and can be found [here](https://huggingface.co/JohannesGaessler/llama.cpp_importance_matrices/blob/main/imatrix-llama_3-8b-f16-2.7m_tokens.dat).

| Quantization | imatrix | Model size [GiB] | PPL               | ΔPPL                 | KLD                 | RMS Δp           |
|--------------|---------|------------------|-------------------|----------------------|---------------------|------------------|
| f16          | None    |            14.97 | 6.7684 ± 0.04278  | -                    | -                   | -                |
| q8_0         | None    |             7.96 | 6.7687 ± 0.04277  | 0.005872 ± 0.001347  | 0.001391 ± 0.000007 | 1.210 ± 0.007 %  |
| q6_K         | None    |             6.14 | 6.8007 ± 0.04312  | 0.037777 ± 0.002294  | 0.005669 ± 0.000046 | 2.343 ± 0.026 %  |
| q5_K_M       | None    |             5.33 | 6.8308 ± 0.04330  | 0.067952 ± 0.003060  | 0.011093 ± 0.000086 | 3.173 ± 0.030 %  |
| q5_K_S       | None    |             5.21 | 6.8877 ± 0.04378  | 0.124777 ± 0.003891  | 0.017177 ± 0.000135 | 3.947 ± 0.037 %  |
| q5_1         | None    |             5.65 | 6.8888 ± 0.04373  | 0.125879 ± 0.004015  | 0.018485 ± 0.000141 | 4.089 ± 0.039 %  |
| q5_0         | None    |             5.21 | 6.8988 ± 0.04373  | 0.135923 ± 0.004525  | 0.022964 ± 0.000170 | 4.631 ± 0.042 %  |
| q4_K_M       | WT 2.7m |             4.58 | 6.9164 ± 0.04390  | 0.153559 ± 0.005115  | 0.029126 ± 0.000256 | 5.270 ± 0.050 %  |
| q4_K_M       | None    |             4.58 | 6.9593 ± 0.04415  | 0.196383 ± 0.005343  | 0.032032 ± 0.000248 | 5.531 ± 0.050 %  |
| q4_K_S       | WT 2.7m |             4.37 | 6.9393 ± 0.04396  | 0.176470 ± 0.005377  | 0.032768 ± 0.000266 | 5.630 ± 0.052 %  |
| iq4_NL       | WT 2.7m |             4.35 | 7.0114 ± 0.04468  | 0.248562 ± 0.005915  | 0.036482 ± 0.000286 | 5.965 ± 0.053 %  |
| iq4_XS       | WT 2.7m |             4.14 | 7.0091 ± 0.04459  | 0.246254 ± 0.005918  | 0.037087 ± 0.000292 | 6.009 ± 0.053 %  |
| q4_K_S       | None    |             4.37 | 7.0545 ± 0.04481  | 0.291578 ± 0.006429  | 0.044040 ± 0.000320 | 6.511 ± 0.055 %  |
| q4_1         | None    |             4.78 | 7.2571 ± 0.04658  | 0.494238 ± 0.009036  | 0.072530 ± 0.000507 | 8.368 ± 0.062 %  |
| q4_0         | None    |             4.34 | 7.2927 ± 0.04665  | 0.529800 ± 0.009048  | 0.073598 ± 0.000486 | 8.395 ± 0.061 %  |
| q3_K_L       | WT 2.7m |             4.03 | 7.2330 ± 0.04666  | 0.470087 ± 0.009268  | 0.074345 ± 0.000530 | 8.577 ± 0.064 %  |
| q3_K_M       | WT 2.7m |             3.74 | 7.2941 ± 0.04699  | 0.531254 ± 0.010144  | 0.085849 ± 0.000596 | 9.236 ± 0.065 %  |
| q3_K_L       | None    |             4.03 | 7.3483 ± 0.04729  | 0.585400 ± 0.010379  | 0.088558 ± 0.000611 | 9.333 ± 0.066 %  |
| q3_K_M       | None    |             3.74 | 7.4524 ± 0.04789  | 0.689517 ± 0.011427  | 0.103797 ± 0.000675 | 10.111 ± 0.068 % |
| iq3_M        | WT 2.7m |             3.53 | 7.5051 ± 0.04715  | 0.742584 ± 0.010752  | 0.104464 ± 0.000676 | 10.383 ± 0.066 % |
| iq3_S        | WT 2.7m |             3.42 | 7.5693 ± 0.04794  | 0.806473 ± 0.011620  | 0.113201 ± 0.000719 | 10.669 ± 0.067 % |
| iq3_XS       | WT 2.7m |             3.28 | 7.8058 ± 0.04967  | 1.042930 ± 0.013767  | 0.140704 ± 0.000846 | 11.979 ± 0.070 % |
| iq3_XXS      | WT 2.7m |             3.05 | 8.0537 ± 0.05169  | 1.290849 ± 0.016815  | 0.187044 ± 0.001042 | 13.722 ± 0.073 % |
| q3_K_S       | WT 2.7m |             3.41 | 8.4003 ± 0.05409  | 1.637409 ± 0.018650  | 0.208394 ± 0.001018 | 15.201 ± 0.070 % |
| q3_K_S       | None    |             3.41 | 8.6701 ± 0.05627  | 1.907244 ± 0.020902  | 0.236401 ± 0.001084 | 15.601 ± 0.069 % |
| iq2_M        | WT 2.7m |             2.74 | 9.4260 ± 0.06254  | 2.663082 ± 0.028667  | 0.331202 ± 0.001611 | 18.368 ± 0.079 % |
| q2_K         | WT 2.7m |             2.96 | 9.4737 ± 0.06303  | 2.710844 ± 0.029119  | 0.342129 ± 0.001565 | 18.996 ± 0.078 % |
| iq2_S        | WT 2.7m |             2.56 | 10.6301 ± 0.07237 | 3.867287 ± 0.039162  | 0.446305 ± 0.001972 | 21.324 ± 0.082 % |
| q2_K         | None    |             2.96 | 10.6450 ± 0.07158 | 3.882171 ± 0.038471  | 0.457258 ± 0.001851 | 21.416 ± 0.078 % |
| iq2_XS       | WT 2.7m |             2.43 | 11.8063 ± 0.08064 | 5.043388 ± 0.048007  | 0.556747 ± 0.002136 | 23.752 ± 0.082 % |
| iq2_XXS      | WT 2.7m |             2.24 | 15.6064 ± 0.11301 | 8.843541 ± 0.081477  | 0.830947 ± 0.002749 | 28.363 ± 0.084 % |
| iq1_M        | WT 2.7m |             2.01 | 28.6561 ± 0.21012 | 21.893176 ± 0.180729 | 1.413517 ± 0.003550 | 37.785 ± 0.084 % |
| iq1_S        | WT 2.7m |             1.88 | 69.6303 ± 0.56051 | 62.867391 ± 0.535295 | 2.290167 ± 0.004882 | 45.826 ± 0.086 % |

## LLaMA 2 vs. LLaMA 3 Quantization comparison

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

| Metric          |         L2 70b q2_K |         L3 70b q2_K |       L2 70b q4_K_M |       L3 70b q4_K_M |         L2 70b q6_K |         L3 70b q6_K |         L2 70b q8_0 |         L3 70b q8_0 |
|-----------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Mean PPL        | 4.172530 ± 0.020805 | 5.902798 ± 0.035278 | 3.475398 ± 0.016580 | 3.193431 ± 0.016621 | 3.440612 ± 0.016372 | 3.052153 ± 0.015746 | 3.434686 ± 0.016346 | 3.039482 ± 0.015687 |
| Mean PPL ratio  | 1.215161 ± 0.002103 | 1.942461 ± 0.007686 | 1.012136 ± 0.000413 | 1.050877 ± 0.001032 | 1.002006 ± 0.000193 | 1.004386 ± 0.000413 | 1.000280 ± 0.000119 | 1.000217 ± 0.000264 |
| Mean ΔPPL       | 0.738805 ± 0.007888 | 2.863974 ± 0.025573 | 0.041672 ± 0.001433 | 0.154607 ± 0.003206 | 0.006887 ± 0.000664 | 0.013329 ± 0.001256 | 0.000961 ± 0.000408 | 0.000658 ± 0.000803 |
| PPL correlation |              93.80% |              75.67% |              99.63% |              98.21% |              99.92% |              99.68% |              99.97% |              99.87% |
| Mean KLD        | 0.186386 ± 0.001134 | 0.674716 ± 0.003267 | 0.013168 ± 0.000095 | 0.055418 ± 0.000506 | 0.002736 ± 0.000018 | 0.009148 ± 0.000100 | 0.000878 ± 0.000006 | 0.003088 ± 0.000040 |
| Mean Δp         |    -5.417 ± 0.040 % |   -17.236 ± 0.078 % |    -0.350 ± 0.010 % |    -1.678 ± 0.026 % |    -0.076 ± 0.005 % |    -0.202 ± 0.010 % |    -0.005 ± 0.003 % |    -0.007 ± 0.006 % |
| Maximum Δp      |             95.064% |             95.799% |             80.018% |             91.140% |             28.193% |             63.263% |             25.395% |             50.187% |
| 99.9% Δp        |             46.526% |             60.640% |             23.562% |             47.583% |             10.424% |             24.634% |              6.548% |             14.033% |
| 99.0% Δp        |             21.251% |             26.948% |             10.161% |             18.666% |              5.339% |             10.273% |              3.337% |              6.323% |
| Median Δp       |             -0.447% |             -3.780% |             -0.004% |             -0.022% |             -0.001% |             -0.002% |             -0.000% |              0.000% |
| 1.0% Δp         |            -81.379% |            -98.506% |            -15.142% |            -47.638% |             -5.866% |            -13.230% |             -3.333% |             -6.609% |
| 0.1% Δp         |            -97.547% |            -99.873% |            -37.914% |            -82.914% |            -13.351% |            -30.683% |             -6.096% |            -15.564% |
| Minimum Δp      |            -99.965% |            -99.993% |            -81.378% |            -98.505% |            -46.213% |            -82.746% |            -34.335% |            -63.634% |
| RMS Δp          |    17.237 ± 0.077 % |    34.361 ± 0.094 % |     4.154 ± 0.032 % |     9.915 ± 0.067 % |     1.899 ± 0.015 % |     3.721 ± 0.030 % |     1.085 ± 0.007 % |     2.124 ± 0.018 % |
| Same top p      |    85.001 ± 0.087 % |    71.991 ± 0.118 % |    95.632 ± 0.050 % |    92.881 ± 0.068 % |    97.651 ± 0.037 % |    96.538 ± 0.048 % |    98.502 ± 0.030 % |    97.825 ± 0.038 % |

## Old Numbers

<details>
<summary>Llama 2 70B Scorechart</summary>

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
