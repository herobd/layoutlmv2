# Fine-tuning LayoutLMv2 on DocVQA with Tesseract evaluation

This code fine-tunes LayoutLMv2 on DocVQA, training from either the dataset provided OCR, or for Tesseract's OCR.
It evaluates using Tesseract's OCR.

This is not as optimized as it could be, but is merely to demonstrate the choice of OCR is quite important for DocVQA.

This was adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb by Brian Davis (hero.bd@gmail.com)

It fixes some issues with aligning the OCR to the GT answers by allowing fuzzy string matching and when an alignment cannot be found it removes the instance, rather than setting the CLS token to be the "answer."

The batch size I used was 5. It validates after an epoch and keeps a snapshot of the best.


Can be fine-tuned using:
`python train_docvqa.py [1/load]` where "1" uses tesseract and "load" uses dataset OCR for training.

Can be run on test dataset with:
`python eval_docvqa.py snapshot.pth output.json`
