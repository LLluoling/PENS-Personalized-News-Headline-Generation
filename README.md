# PENS-Personalized-News-Headline-Generation
PENS - ACL2021 

Code for PENS: A Dataset and Generic Framework for Personalized News Headline Generation

This is a Pytorch implementation of [PENS](https://www.microsoft.com/en-us/research/uploads/prod/2021/06/ACL2021_PENS_Camera_Ready_1862_Paper.pdf). 


## 0. Enviroment
- Install pytorch version >= '1.4.0'
- Install the pensmodule package under ''PENS-Personalized-News-Headline-Generation'' using code ``` pip install -e . ```

## 1. Data Prepare
- Download the PENS dataset [here](https://msnews.github.io/pens.html) and put the dataset under data/.
- (optional) Download glove.840B.300d.txt under data/ if you choose to use pretrained glove word embeddings.

### 2. Running Code
- ```cd pensmodule ```
- Follow the order: Preprocess --> UserEncoder --> Generator and run the **.ipynb notebook to preprocess, train the user encoder and the train generator, individually.

### 3. Running Code
- ```cd pensmodule ```
- Follow the order: Preprocess --> UserEncoder --> Generator and run the pipeline**.ipynb notebook to preprocess, train the user encoder and the train generator, individually.

More infor please refer to the homepage of the [introduction of PENS dataset](https://msnews.github.io/pens.html).
