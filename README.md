# From Voice to Safety: Language AI Powered Pilot-ATC Communication Understanding for Airport Surface Movement Collision Risk Assessment.
### 2024 FAA Data challenge of the UT Austin Team
**Advised by Prof. John-Paul Clarke**

This repository implements a rule-enhanced automatical speech recognition for air traffic communication transcript understanding, and build a collision risk assessment model from it, using the learned destination node similarities from NASA FACET airport node-link graphs.  
## Motivation
The United States is currently facing an alarming escalation in aviation safety incidents. In the first six weeks of 2025 alone, there have been over 30 commercial aviation incidents/accidents, including four major plane crashes that have tragically claimed 85 lives. 
![Animation](faa-accidents-hist.png)

Accidents and Incidents in the continental U.S. from April 2024 to February 2025, documented by the FAA:
1. [Link to Figure Visualization](faa_events_map.html) 
2. [Link to FAA](https://www.faa.gov/newsroom/statements/accident_incidents)

## Methodology Overview
![Animation](ATCTranscriptLearningNLP.png)

# Project Overview

This project involves multiple Jupyter notebook files that work together to prepare data, train machine learning models, and visualize the results.

## Notebook Descriptions
- **`ner_data_preparation.ipynb`**: Contains the code for data preparation including the usage of Whisper AI and Tesseract.
- **`ner_transformer.ipynb`**: Includes the code for training the machine learning model with transformer embeddings.
- **`ner_tok2vec.ipynb`**: Includes the code for training the machine learning model with tok2vec embeddings.
- **`results_visualization.ipynb`**: Visualizes the results of the machine learning model.
- **`case_study_1_demo.ipynb`**: Generates all the simulation steps for Case Study 1.
- **`case_study_2_demo.ipynb`**: Generates all the simulation steps for Case Study 2.

## Travel Time Modeling:
The $k$-th aircraft travels a total of $n$ taxiway links until reaching the certain spot of interest (i.e., potential collision spot), where the total travel time is given by $\Gamma_k$. We assume each taxiway link has an associated distance $d_{k,i}$ and a taxi speed $v_{k,i}$ that is log-normally distributed with parameters $\mu_{k,i}$ and $\sigma_{k,i}^2$, which is

$$
v_{k,i} \sim \mathrm{Lognormal}(\mu_{k,i}, \sigma^2_{k,i}).
$$


It is obvious that $\Gamma_k = \sum_{i=0}^n \tau_{k,i}$ where $\tau_{k,i} = \tfrac{d_{k,i}}{v_{k,i}}$ is the distribution of the $k$-th aircraft travel time duration for the $i$-th node link. By the standard formula for transformations of random variables, if $\tau_{k,i} = g(v_{k,i}) = \tfrac{d_{k,i}}{v_{k,i}}$, then:

$$
f_{\tau_{k,i}}(\tau_{k,i}) =
f_{v_{k,i}}\bigl(g^{-1}(\tau_{k,i})\bigr)\;\left|\frac{d}{d\tau_{k,i}}\,g^{-1}(\tau_{k,i})\right|.
$$

That is, each $\tau_{k,i}$ is a lognormal-type variable, with parameters shifted by $\ln d_{k,i}$:

$$
\tau_{k,i} \sim \mathrm{Lognormal}\bigl(\ln d_{k,i} - \mu_{k,i}, \sigma^2_{k,i}\bigr).
$$

with

$$
\mathbb{E}[\tau_{k,i}] = d_{k,i}\,\exp\!\Bigl[-\mu_{k,i} + \tfrac{\sigma_{k,i}^2}{2}\Bigr],
\quad
Var[\tau_{k,i}] = d_{k,i}^2\,\exp\!\Bigl(-2\mu_{k,i} + \sigma_{k,i}^2\Bigr)\bigl[\exp\!\bigl(\sigma_{k,i}^2\bigr) - 1\bigr].
$$

The total travel time for the $k$-th aircraft, $\Gamma_k$, is the $n$-fold convolution of each individual link distribution.

In practice, we approximate $f_{\Gamma_k}(t_k)$ for any time $t_k>0$ by either Monte Carlo Simulations or Moment-Matching Approximations. For the convolution of log-normal distributions with moderate variance and $n_k$, the Fenton-Wilkinson approach provides a feasible solution to directly match the first two moments, and is widely adopted as the approximated analytical solution of log-normal sums in various fields.

That is, we look for parameters of an approximate distribution $\Gamma_k \approx X_k^*$ where

$$
X_k^* \sim \mathrm{Lognormal}(\mu_k^*, \sigma_k^{*2}).
$$

where

$$
\mu_k^* = \ln M_k - \tfrac12 \ln(1 + \tfrac{V_k}{M_k^2}),
\quad
\sigma_k^{*2} = \ln(1 + \tfrac{V_k}{M_k^2}).
$$

with

$$
M_k = \sum_{i=1}^{n_k} \mathbb{E}[\tau_{k,i}],
\quad
V_k = \sum_{i=1}^{n_k} \mathrm{Var}[\tau_{k,i}].
$$


  Each taxiway link has an associated distance (computed from latitude and longitude using the haversine formula) and speed parameters (mu and sigma) determined by the link type. For example:
  - If both nodes are of type `Rwy` (Runway), then the speed parameters are `(30, 10)`.
  - If one node is `Rwy` and the other is `Txy` (Taxiway), then the parameters are `(25, 5)`.
  - If both nodes are `Txy`, then the parameters are `(20, 5)`.
  - For other node types (e.g., `Ramp`, `Gate`), the default parameters are `(15, 5)`.


**Collision Risk Calculation:**  
The collision happens when the two aircraft arrive simultaneously at the potential collision spot from the airport node-link graph (i.e., \(\Gamma_1 = \Gamma_2\) theoretically). Thus, the collision risk is measured by the density of the difference:

$$
\digamma = \Gamma_1 - \Gamma_2.
$$

Evaluated at \(\digamma=0\), this is given by:

$$
f_\digamma(\digamma = 0) = \int_0^\infty f_{\Gamma_1}(t)\, f_{\Gamma_2}(t)\, dt.
$$



### Case Study I: 2024 Henada Airport Runway Incursion
#### ATC Rule-enhanced ASR Results

| **TIME**  | **CALLSIGN**     | **ACSTATE**        | **DEST_RUNWAY** | **DESTINATION**        |
|-----------|------------------|--------------------|-----------------|------------------------|
| 17:43:02  | Japan Air 516     | approach, departure | 34R             | Rwy_03_001             |
| 17:43:12  | Japan Air 516     | approach           | 34R             | Rwy_03_001             |
| 17:43:26  | Delta 276         | taxi               | 34R             | Txy_C1_C (holding point C1) |
| 17:44:56  | Japan Air 516     | cleared, land      | 34R             | Rwy_03_001             |
| 17:45:01  | Japan Air 516     | cleared, land      | 34R             | Rwy_03_001             |
| 17:45:11  | JA722A            | taxi               |                 | Txy_C5_C5B (holding point C5) |
| 17:45:19  | JA722A            | taxi               |                 | Txy_C5_C5B (holding point C5) |
| 17:45:40  | Japan Air 179     | taxi               |                 | Txy_C1_C (holding point C1) |
| 17:45:56  | Japan Air 166     | approach           | 34R             | Rwy_03_001             |
| 17:47:23  | Japan Air 166     | approach           | 34R             |                        |
| 17:47:27  | Japan Air 166     |                    | 34R             |                        |
| 17:47:30  | Japan Air 516     | collision          |                 |                        |
| 17:47:30  | JA722A            | collision          |                 |                        |

#### Collision Risk Assessment
![Animation](./taxigen/case-study-1-riskmap.gif)





### Case Study II: 2024 KATL Taxiway Collision
#### ATC Rule-enhanced ASR Results

| **CALLSIGN**    | **TIME**  | **AC_STATE**          | **DEST_RUNWAY** | **DESTINATION**        |
|-----------------|-----------|-----------------------|-----------------|------------------------|
| Delta 295       | 0:08      | taxi                  | 08R             | Romeo                  |
| Delta 295       | 0:14      | taxi                  | 08R             | Rwy_02_001             |
| Delta 295       | 0:20      | Taxi                  | 08R             | foxtrot                |
| Delta 295       | 0:33      | continue, hold        | 08R             | ramp 5                 |
| Delta 295       | 0:44      | give way, inbound, join | 08R           | Echo                   |
| Delta 295       | 0:50      | give way              | 08R             |                        |
| Endeavor 5526   | 0:57      | taxi                  | 08R             | Rwy_02_001             |
| Delta 295       | 1:27      | go                    | 08R             |                        |
| Delta 295       | 1:35      | continue, hold        | 08R             |                        |
| Delta 295       | 1:45      | holding               | 08R             | Victor                 |
| Endeavor 5526   | 1:54      | line up, wait         | 08R             |                        |
| Endeavor 5526   | 2:10      | collision             |                 |                        |
| Delta 295       | 2:10      | collision             |                 |                        |


#### Collision Risk Assessment
![Animation](./taxigen/case-study-2-riskmap.gif)


# Citations
If you find this work useful in your research, please cite us,
```
@article{}

```