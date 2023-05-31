# OTExtSum

This code is for paper [OTExtSum: Extractive Text Summarisation with Optimal Transport](https://aclanthology.org/2022.findings-naacl.85/), Findings of NAACL 2022

### To test the performance with BIP optimisation strategy using BERT representation:
> python OTExtSum_BIP.py --save_path=decoded_bip_cnndm3 --dataset_str cnn_dailymail --dataset_doc_field article --max_output_length 3 --tkner bert --device cuda



### To test the performance with BS optimisation strategy using BERT representation:
> python OTExtSum_BS.py --save_path=decoded_bs_pubmed6 --dataset_str pubmed --dataset_doc_field article --tkner bert --max_output_length 6 --device cuda
