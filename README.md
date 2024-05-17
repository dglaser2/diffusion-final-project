# Predicting Stable Diffusion Prompts Based on Generated Images

This repository contains our experiments for the Kaggle competition -> [Stable Diffusion Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts). 

## Methodology

### Datasets

1. **SDDBM2**: This dataset consists of 2 million images produced by Stable Diffusion along with their text prompts. Due to hard drive constraints, we trained on 1000 of these image-prompt pairs, which resulted in bizarre outputs. We believe this is due to the small dataset size and the fragmented nature of the prompts, many of which are simply lists of unlikely pairings of adjectives and nouns. We hypothesize that training on a larger dataset with more epochs could improve the results.

2. **COCO**: This dataset contains about 80,000 image-caption pairs, of which we used 2000. We chose COCO because its captions have a more typical sentence structure and hoped this would facilitate faster fine-tuning of our pre-trained GPT-2 model.

### Models

1. **BLIP-2 Architecture**:
    - **Image Encoder**: We used OpenAI’s pretrained “clip-vit-base-patch32” model. The CLIP model embeds images such that they align in a projected space with the embeddings of their textual descriptions.
    - **Transformer**: The image embedding from the CLIP model is passed into a transformer, generating tokens that represent the image’s most prominent features.
    - **GPT-2**: The tokens are then input into a pre-trained GPT-2 model to generate a coherent caption or prompt for the image.


    During training, we froze the CLIP model and all but the last two layers of the GPT-2 model. Our training focused on teaching the bridging transformer and fine-tuning GPT-2 to output sentences similar to those in our diffusion prompt and caption datasets. We calculated the loss using negative log-likelihood.

    Due to memory constraints in Kaggle, we experimented with batch sizes of 256, 128, and 64, eventually training successfully with a batch size of 64. We also increased the number of epochs from 30 to 50, expecting results to improve with more epochs.

2. **CLIP Interrogator**:
    - We used a pretrained CLIP model (ViT-H-14) and a pretrained BLIP model (model_large_caption) along with Sentence Transformers for text embeddings.
    - Instead of traditional matrix multiplication for determining similarity between image features and text embeddings, we applied cosine similarity, which accelerated the process while yielding nearly identical scores.
    - We developed a custom interrogator function using the pretrained models to generate descriptive prompts from images, enhancing the efficiency and accuracy of producing contextually relevant captions.

3. **OFA (One For All) Model**:
    - Inputs: Images and the textual query, "What does the image describe?"
    - Text was tokenized using the OFA Tokenizer, and Sentence Transformers were used for embedding generation and submission data processing.
    - This approach leveraged the robust capabilities of the OFA model to generate comprehensive descriptions, enriching our comparative analysis of various model performances.


## Notebooks

The notebooks are organized as follows:

1. **images-to-text-coco.ipynb**: Notebook for OpenAI’s pretrained “clip-vit-base-patch32” model with GPT-2 and MS COCO dataset
2. **images-to-text-sddbm2.ipynb**: Notebook for OpenAI’s pretrained “clip-vit-base-patch32” model with GPT-2 and SDDBM dataset
3. **CLIP-interrogator**: Notebook with the BLIP, CLIP and the CLIP interrogator
4. **ofa-transformer.ipynb**: Notebook for the OFA model

## References

- [SDDBM2 Dataset]()
- [COCO Dataset](https://cocodataset.org/#home)
- [BLIP-2 Architecture](https://arxiv.org/pdf/2301.12597)
- [OpenAI CLIP Model](https://arxiv.org/pdf/2103.00020)
- [Sentence Transformers](https://www.kaggle.com/datasets/inversion/sentence-transformers-222)
- [OFA Model](https://arxiv.org/pdf/2202.03052)
- [Kaggle Notebook OFA](https://www.kaggle.com/code/mayukh18/ofa-transformer-lb-0-4264)
- [Kaggle Notebook BLIP/CLIP](https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator)
