---
title: 'Accurate Unsupervised Photon Counting from Transition Edge Sensor Signals'
tags:
  - Tag1
  - Tag2
authors:
  - name: Nicolas Dalbec-Constant
    email: nicolas.dalbec-constant@polymtl.ca
    affiliation: 1
  - name: Guillaume Thekkadath
    affiliation: 2
  - name: Duncan England
    affiliation: 2
  - name: Benjamin Sussman
    affiliation: 2
  - name: Thomas Gerrits
    affiliation: 3
  - name: Nicolás Quesada
    email: nicolas.quesada@polymtl.ca
    affiliation: 1
affiliations:
  - name: D\'epartement de g\'enie physique, \'Ecole Polytechnique de Montr\'eal, Montr\'eal, QC, H3T 1J4, Canada
    index: 1
  - name: National Research Council of Canada, 100 Sussex Drive, Ottawa, Ontario K1N 5A2, Canada
    index: 2
  - name: National Institute of Standards and~Technology, 100 Bureau Drive, Gaithersburg, MD 20899, USA
    index: 3
date: 2 november 2024
bibliography: paper.bib
---

# Abstract

We compare methods for signal classification applied to voltage traces from transition edge sensors (TES) which are photon-number resolving detectors fundamental for accessing quantum advantages in information processing, communication and metrology. We quantify the impact of numerical analysis on the distinction of such signals. Furthermore, we explore dimensionality reduction techniques to create interpretable and precise photon number embeddings. We demonstrate that the preservation of local data structures of some nonlinear methods is an accurate way to achieve unsupervised classification of TES traces. We do so by considering a confidence metric that quantifies the overlap of the photon number clusters inside a latent space. Furthermore, we demonstrate that for our dataset previous methods such as the signal's area and principal component analysis can resolve up to 16 photons with confidence above $90\%$ while nonlinear techniques can resolve up to 21 with the same confidence threshold. Also, we showcase implementations of neural networks to leverage information within local structures, aiming to increase confidence in assigning photon numbers. Finally, we demonstrate the advantage of some nonlinear methods to detect and remove outlier signals.

# Introduction

Photonics is a strong contender for building large-scale quantum information processing systems [@arrazola_quantum_2021,@slussarenko_photonic_2019,@rudolph2017optimistic,@bourassa2021blueprint,@maring2024versatile]; in many of these systems, photon number detection plays an essential role, serving as a resource for quantum advantage. These detectors can be used, for example, for the heralded generation of non-Gaussian states [@takase2024generation,@alexander2024manufacturable,@yao2022design,@chen2024generation,@melalkia2023multiplexed,@tiedau2019scalability,@sonoyama2024generation,@endo2024optically], for the sampling of classically-intractable probability distributions [@aaronson2011computational,@hamilton2017gaussian,@kruse2017limits,@deshpande2022quantum,@grier2022complexity,@madsen2022quantum] or for directly resolving multiple quanta improving the Fisher information of interferometric protocols [@thekkadath2020quantum,@Wildfeuer:09]. The use of photon number resolving detectors provides a significant advantage as a single detector can determine the number of photons associated with a quantum state accurately [@divochiy_superconducting_2008,@moraisPreciselyDeterminingPhotonnumber2022a], without requiring a multiplexed network of threshold detectors with its concomitant complexity and potential inefficiency[@kruse2017limits,@jonsson2019evaluating,@jonsson2020temporal]. Transition edge sensors (TES) have been used for this task, offering resolution over a wide energy range. Resolutions up to 30 photons have been demonstrated [@eaton2023], although typically this quantity is typically lower, on the order of 17, if more straightforward techniques are used [@moraisPreciselyDeterminingPhotonnumber2022a]. 

TESs exploit the superconducting phase transition of photosensitive materials to achieve an extremely sensitive calorimeter [@irwin_transition-edge_2005]. During operation, the material is cooled below its critical temperature and then current-biased to the transition region between its superconducting and normal state. In this region, the temperature increase following the absorption of a single photon leads to a measurable change in the material's resistance [@phillips2020advanced,@hadfield2009single]. The resistance change is read-out using a low noise amplifier such as superconducting quantum interference devices (SQUIDs), which also enable the creation of large arrays of TES detectors via read-out multiplexing [@irwin_transition-edge_2005]. Optimized materials and coupling techniques have demonstrated efficiencies of up to 98\% [@fukuda_titanium-based_2011]. 

The readout of these devices is non-trivial as the quantity one wants to determine, the energy (or the photon number for a fixed frequency), is reflected in a nonlinear fashion in the voltage signal produced by the detectors' electronics [@gerrits_extending_2012]. Historically, the integral (area) of the signals has been used to assign photon numbers [@moraisPreciselyDeterminingPhotonnumber2022a,@Schmidt_Bimodal_2021]. However, distinguishing large photon numbers becomes challenging with this technique. To address this issue, linear techniques such as Principal Component Analysis (PCA) have been used [@humphreys_tomography_2015]. A machine learning method, adapted from the K-means algorithm to account for the Poissonian statistics of laser sources, has also been developed [@levine_algorithm_2012]. However, these methods' simplicity or assumptions can limit their performance or usability for model-free photon number detection and when measuring non-classical sources, which typically do not have Poisson photon-number statistics.

With the increased popularity of machine learning in the field of signal processing [@rajendran_deep_2018] and quantum systems [@nautrup_optimizing_2019], one might naturally ask whether employing more sophisticated methods could lead to enhanced resolution of photon numbers. In this work, we answer this question by assessing the performance of multiple techniques for photon number classification using TES signals. We do so by considering a performance metric named confidence that quantifies the overlap of the signal's probability distribution inside an embedding. We demonstrate that for our dataset previous methods such as the signal's area and PCA can resolve up to 16 photons with confidence above $90\%$ while nonlinear techniques can resolve up to 21 with the same confidence threshold. Furthermore, we also showcase implementations of neural networks to leverage information within local structures, aiming to increase confidence in assigning photon numbers. Finally, we demonstrate the advantage of some nonlinear methods to detect and remove outlier signals. 

# Clustering

Clustering refers to identifying groups of similar samples inside a latent space. For this task we use a Gaussian mixture model, given a user-defined number of clusters, this method finds the parameters of a mixture of Gaussians to describe the sample's distribution.

The choice is highly inspired by a similar model previously used in the tomography of TESs in combination with PCA [@humphreys_tomography_2015]. Mixture models offer a statistical interpretation of latent spaces convenient for metrology and performance evaluation. 

The mixture model gives a continuous probability density function for the position $s$ of samples given optimal parameters $\theta=\{(\omega_k, \mu_k$, $\Sigma_k):k=1,\cdots,K\}$. In the model, every cluster $k$ is weighted by a value $\omega_k$ (where $\sum_{k=1}^K \omega_k = 1$), and modelled by a Gaussian with mean $\mu_k$ and covariance matrices $\Sigma_k$. The individual Gaussians $\mathcal{N}$ give the cluster probability density function and the probability of observing samples in position $s$ given parameters $\theta$ are defined by

$$
p(s|\theta) = \sum_{k=1}^K \omega_k \mathcal{N}(s|\mu_k , \Sigma_k).
$$

The probability density function is found through an expectation maximization algorithm (EM algorithm) that attempts to find the maximum likelihood estimate of samples following a likelihood of  

$$
\mathcal{L}(\theta) = \prod_{i=1}^p \sum_{k=1}^K \omega_k \mathcal{N}(s_i|\mu_k , \Sigma_k).
$$

Numerically it is more convenient to express this problem in terms of the log-likelihood given by 

$$
\ell(\theta) = \log(\mathcal{L}(\theta)) = \sum_{i=1}^p \log\left(\sum_{k=1}^K \omega_k \mathcal{N}(s_i|\mu_k , \Sigma_k)\right),
$$

where the problem can be computed in terms of sum instead of products.


# Quality Assessment

Assessing the performance of dimensionality reduction techniques in an unsupervised setting is difficult since the ground truth is unknown. To tackle this task, we quantify cluster separation. To improve the performance evaluation it is also important to understand that the problem is not completely unsupervised considering photon sources used to generate samples follow known distributions. We include this knowledge of photon number distributions as an additional validation to cluster separation evaluation in the confidence metric. 


## Confidence

We consider the probability density of photon events can be approximated from the sample's distribution in the latent space following the Gaussian mixture model. Following previous work [@humphreys_tomography_2015], the confidence $C_n$ is used as a performance metric for the resolution of photon numbers in a latent space following,

$$
C_n = \int_{-\infty}^{\infty} \frac{p(s|n)^2 P(n)}{\sum_k p(s|k) P(k)} \mathrm{d} s.
$$

In this equation, $p(s|n)$ is the probability density of observing a sample in position $s$ in the latent space given it is labelled as $n$ photons. Additionally, $P(n)$ is the probability of assigning a photon number $n$. In this model, we consider that the true clusters follow a Gaussian structure inside the latent space. The confidence represents the probability of correctly labelling a sample in a given cluster in the mixture model. We note that that confidence equation describes the confidence for a one-dimensional space but can be generalized to an arbitrarily high-dimensional latent space. It is important to mention that the distances in the latent space do not necessarily have a physical meaning. The separation must only be interpreted as our capacity to distinguish clusters, and the confidence translates this concept into a probabilistic framework.


## Datasets

Experimental data from previous work at the National Institute of Standards and Technologies (NIST) is used to benchmark the different techniques in this work [@gerrits_extending_2012]. The original dataset was generated by progressively attenuating a coherent source from 29dB to 7dB, leading to 24 datasets each containing $u=20\,480$ signals and $t=8\,192$ time steps. This results in datasets that each have Poisson photon number distributions and mean photon number $\langle n_1 \rangle=2.26$ to $\langle n_{24} \rangle=7.08\times 10^6$. These values were independently measured using a calibrated photodetector. 

Instead of directly using these distributions, we construct two synthetic datasets (made of real traces) that follow a close-to-uniform and close-to-geometric distributions $P(n)$. These datasets are labelled as Synthetic Uniform and Synthetic Geometric. Furthermore, for both of these datasets, a training and testing set were generated. Considering randomly selecting a portion of the samples in each experiment is equivalent to varying the weight $w_{{\langle n \rangle}}$ of a given Poisson distribution $P_{{\langle n \rangle}}(n)$ inside a mixture of Poisson distributions. The total expected distribution $P(n)$ can be described by

$$
P(n) = \frac{1}{\xi} \sum_{\langle n \rangle\in \bar{N}} w_{\langle n \rangle} P_{\langle n \rangle}(n),
$$

with

$$
\xi = \sum_{\langle n \rangle \in \bar{N}} w_{\langle n \rangle},
$$

and where $\bar{N}$ is the set of available mean photon numbers $\langle n \rangle$. With this construction, the expected photon number distribution is a mixture of Poisson distributions. The choice of a uniform distribution is motivated by the desire to make the labelling task difficult by maximizing the distribution's entropy. In other words, for every sample in a perfectly uniform distribution, the method would have equal chances of guessing every class. The choice of testing a geometric distribution comes from the desire to precisely measure thermal optical sources that follow a geometric photon number distribution. Also, distributions with a long tail can be difficult to process for certain methods since fewer examples are present in some classes (imbalanced dataset).

We add that these expected distributions are used as $P(n)$ in the computation of the confidence. The predictive methods are trained with the training set, and the analysis of performance metrics is done by feeding the test set to the trained methods. In the case of non-predictive and basic feature methods, the test set is directly used. The training and test datasets contain a total of $u=30\,550$ traces of size $t=350$ (first $350$ values of the $8192$ available time steps). We note that most of the weights $w_{\langle n \rangle}$ are set to zero because of the number of available Poisson distributions in the desired photon number range is small, making the synthetic distribution not perfectly uniform.

To validate the hypothesis that more training data can help parametric implementations of t-SNE and UMAP resemble there non-parametric equivalent, we also use a larger dataset named Synthetic Large that was created using signals generated by TESs at the National Research Council Canada (NRC) in Ottawa. The data was generated by tuning the attenuation of a laser and measuring $u=100 \,000$ signals for each of these coherent sources.

# References

<!-- ```{bibliography}
:style: unsrt
``` -->