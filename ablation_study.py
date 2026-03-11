from run_ablation import run_ablation_study
from test_ablation import test_ablation
from make_gif import make_gifs_for_study
import time

def run_full_pipeline():
    """
    Executes the complete ablation study pipeline:
    1. Trains all configurations over multiple seeds.
    2. Tests the best model from each configuration and generates reports.
    3. Creates comparison GIFs for visual analysis.
    """
    t_start = time.time()
    run_time = time.time()
    print("--- Starting Full Ablation Study Pipeline ---")

    print("\n[Step 1/3] Running Ablation Training...")
    run_ablation_study()
    print("\n[Step 1/3] Finished Ablation Training.")
    print(f'Ablation Training time: {(time.time() - run_time)/60:.0f} minutes = {(time.time() - run_time)/60/60:.1f} hours')
    run_time = time.time()

    print("\n[Step 2/3] Running Ablation Testing and Reporting...")
    test_ablation()
    print("\n[Step 2/3] Finished Ablation Testing.")
    print(f'Ablation Testing time: {(time.time() - run_time)/60:.0f} minutes = {(time.time() - run_time)/60/60:.1f} hours')
    run_time = time.time()

    print("\n[Step 3/3] Generating Comparison GIFs...")
    make_gifs_for_study()
    print("\n[Step 3/3] Finished Generating GIFs.")
    print(f'Ablation GIF generation time: {(time.time() - run_time)/60:.0f} minutes = {(time.time() - run_time)/60/60:.1f} hours')
    run_time = time.time()

    print("\n--- Full Ablation Study Pipeline Complete! ---")
    print(f'Total time taken: {(time.time() - t_start)/60:.0f} minutes = {(time.time() - t_start)/60/60:.1f} hours')

if __name__ == '__main__':
    run_full_pipeline()