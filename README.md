## How to Run
Go into each example/ and run: python main.py --train=1 to train; python main.py --train=0 to evaluate and visualize using pre-trained networks.

## Structure
Each example/ has outputs/eval_bundle.pth that saves all the objects, and outputs/terminal_log.txt that log the training.
Each example/ has results/ that summarize the last epoch of the training

## Notes
The inv_pend_veri has not been SAT yet.

## Comparison
To run the comparison, go into third_party/sumi-lab, run pip install -r requirements.txt, then run: python run_gbm.py or run_gbm3d.py. The results would be saved as a .txt file in the sumi-lab/ folder.
The code may be killed (a sudden interuption in the .txt) if too many cells are being refined.
