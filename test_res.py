import argparse
import torch
import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance
import os 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def process_audio(path, solver, nfe, tau, denoising):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    tau = float(tau)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    return (new_sr, wav1), (new_sr, wav2)

def process_and_save_audio(path, solver, nfe, tau, denoising):
    if path is None:
        return None

    solver = solver.lower()
    nfe = int(nfe)
    tau = float(tau)
    lambd = 0.9 if denoising else 0.1

    # Load audio
    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    # Process audio
    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    # Save audio files
    base_name = os.path.splitext(os.path.basename(path))[0]
    denoised_path = f"{base_name}_denoised.wav"
    enhanced_path = f"{base_name}_enhanced.wav"

    torchaudio.save(denoised_path, wav1.unsqueeze(0), new_sr)
    torchaudio.save(enhanced_path, wav2.unsqueeze(0), new_sr)

    return denoised_path, enhanced_path


'''
# python test_denoise.py path_to_audio_file --solver midpoint --nfe 128 --tau 0.5 --denoising

def main():
    parser = argparse.ArgumentParser(description="AI-driven audio enhancement for your audio files.")
    parser.add_argument("file_path", type=str, help="Path to the input audio file")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "rk4", "euler"], help="CFM ODE Solver")
    parser.add_argument("--nfe", type=int, default=64, help="Number of Function Evaluations")
    parser.add_argument("--tau", type=float, default=0.5, help="Prior Temperature")
    parser.add_argument("--denoising", action="store_true", help="Apply denoising before enhancement")

    args = parser.parse_args()

    # Process audio file
    results = process_audio(args.file_path, args.solver, args.nfe, args.tau, args.denoising)

    # Print or save results as needed
    if results is not None:
        print("Audio processing complete.")
        # Additional code to save or further process the outputs can be added here.

if __name__ == "__main__":
    main()
'''

# Define the path and parameters directly in the script
file_path = "test1_audo.mp3"  # Update the path to your audio file
solver = "midpoint"  # Solver can be 'midpoint', 'rk4', or 'euler'
nfe = 128  # Number of Function Evaluations
tau = 0.5  # Prior Temperature
denoising = True  # Whether to apply denoising before enhancement

'''
results = process_audio(file_path, solver, nfe, tau, denoising)

   # Print or save results as needed
if results is not None:
    print("Audio processing complete.")
    # Additional code to save or further process the outputs can be added here.

'''

# Process and save audio file
denoised_file, enhanced_file = process_and_save_audio(file_path, solver, nfe, tau, denoising)

if denoised_file and enhanced_file:
    print(f"Audio processing complete. Files saved as {denoised_file} and {enhanced_file}.")

