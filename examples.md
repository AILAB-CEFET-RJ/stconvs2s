## Examples of how to run the experiments

### STConvS2S-R model
------- Dataset: CFSR ------------- 

------- Step: 5 ------------------- 

'num_layers': 3, 'kernel_size': 5, 'hidden_dim': 32

`python main.py -i 3 -c 0 -v 4 -l 3 -d 32 -k 5 -m stconvs2s-r --plot --email > output/full-dataset/results/stconvs2s-r/cfsr-stconvs2s-r-step5-v4.out`

------- Dataset: CHIRPS ----------- 

------- Step: 15 ------------------ 

'num_layers': 2, 'kernel_size': 5, 'hidden_dim': 8

`python main.py -i 3 -c 0 -v 4 -l 2 -d 8 -k 5 -m stconvs2s-r --plot --email --chirps -s 15 > output/full-dataset/results/stconvs2s-r/chirps-stconvs2s-r-step15-v4.out`


### Ablation study
------- Model:   STConvS2S-NoCausalConstraint ------ 

------- Dataset: CFSR ------------------------------ 

'num_layers': 3, 'kernel_size': 5, 'hidden_dim': 32

`python main.py -i 1 -c 0 -v 4 -l 3 -d 32 -k 5 -m ablation-stconvs2s-nocausalconstraint --plot --email --no-stop -e 50 > output/full-dataset/results/ablation-stconvs2s-nocausalconstraint/cfsr-ablation-stconvs2s-nocausalconstraint-step5-v4.out`


