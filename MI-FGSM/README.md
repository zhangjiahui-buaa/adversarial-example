This is a simple implementation of the paper **Boosting Adversarial Attacks with Momentum**

part of the result of table 1 in the paper

|           | Attack method | inc_v3   |          | inc_v4   |          | incres_v2 |          |
| --------- | ------------- | -------- | -------- | -------- | -------- | --------- | -------- |
| inc_v3    | FGSM          | **72.3** | **76.8** | 28.2     | 31.2     | 26.2      | 29.5     |
|           | I-FGSM        | **100**  | **99.9** | 22.8     | 22.8     | 19.9      | 21.5     |
|           | MI_FGSM       | **100**  | **99.9** | 48.8     | 48.8     | 48.0      | 47.7     |
| inc_v4    | FGSM          | 32.7     | 30.1     | **61.0** | **50.8** | 26.6      | 21.3     |
|           | I_FGSM        | 35.8     | 19.0     | **99.9** | **99.6** | 24.7      | 22.0     |
|           | MI_FGSM       | 65.6     | 49.6     | **99.9** | **99.6** | 54.9      | 45.8     |
| incres_v2 | FGSM          | 32.6     | 26.3     | 28.1     | 19.6     | **55.3**  | **43.2** |
|           | I-FGSM        | 37.8     | 20.6     | 20.8     | 24.6     | **99.6**  | **97.4** |
|           | MI-FGSM       | 69.8     | 52.7     | 62.1     | 50.4     | **99.5**  | **97.8** |

the definition of the table is:

​	take the first-four rows as example. We generate adversarial example on inc_v3 using three attack methods and we use the adversarial example to perform black-box attack on inc_v4 and incres_v2. The left side of each result is from the original paper, and the right side of each result is from my reproduction. 

​	It can be observed that the absolute number is not as good as that in the paper. However, the trend that MI-FGSM works best in black-box attack can be proved by my reproduction.