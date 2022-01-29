# MLII - NCF

Project in Advanced Machine Learning at the University of Twente. 

## Are VAE/AE replaceable in Recommendation System?

## Recommending content taking into account the appreciation of the user

## Planning 
* Week 50: Delivery of the project's proposal.
* Weeks 51: Encode the dataset and adapt NCF implementation.
* Weeks 2 - 3: Implement NEAT into the NCF structure.
* Week 3: Optimize NEAT and evaluate the model on the dataset encodings.
* Week 4: Presentation of project.
* Week 5: Delivery of project report.

## Requirements
code functioning on conda enviroment with:
python                    3.7.11
tensorflow                1.14.0
numpy                     1.21.5
pip                       21.2.4

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run GMF:
```
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run MLP:
```
python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run NeuMF (without pre-training): 
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run NeuMF (with pre-training):
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5
```

Note on tuning NeuMF: our experience is that for small predictive factors, running NeuMF without pre-training can achieve better performance than GMF and MLP. For large predictive factors, pre-training NeuMF can yield better performance (may need tune regularization for GMF and MLP). 
