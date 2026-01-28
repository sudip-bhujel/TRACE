# uv run -m ppo.capture_gradients

# Efficient Gradient Capture for PPO Agent

Loading model from: checkpoints/ppo_ai2thor_final.pt
Loaded checkpoint from update 500
Global steps: 128000
Episodes: 1984

Initializing AI2-THOR environment...

Capturing 100 trajectories...
Gradient size: 3,356,646 values (6555.9 KB per step as float16)
Estimated total steps: ~10,000
Created HDF5 file: trajectory_data/gradients.h5
Trajectory 1/100 | Steps: 15 | Reward: 0.85 | Success: ✓ | Total: 15 steps
Trajectory 2/100 | Steps: 4 | Reward: 0.96 | Success: ✓ | Total: 19 steps
Trajectory 3/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 219 steps
Trajectory 4/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 419 steps
Trajectory 5/100 | Steps: 11 | Reward: 0.89 | Success: ✓ | Total: 430 steps
Trajectory 6/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 462 steps
Trajectory 7/100 | Steps: 21 | Reward: 0.79 | Success: ✓ | Total: 483 steps
Trajectory 8/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 507 steps
Trajectory 9/100 | Steps: 45 | Reward: 0.55 | Success: ✓ | Total: 552 steps
Trajectory 10/100 | Steps: 19 | Reward: 0.81 | Success: ✓ | Total: 571 steps
Trajectory 11/100 | Steps: 31 | Reward: 0.69 | Success: ✓ | Total: 602 steps
Trajectory 12/100 | Steps: 37 | Reward: 0.63 | Success: ✓ | Total: 639 steps
Trajectory 13/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 663 steps
Trajectory 14/100 | Steps: 185 | Reward: -0.85 | Success: ✓ | Total: 848 steps
Trajectory 15/100 | Steps: 18 | Reward: 0.82 | Success: ✓ | Total: 866 steps
Trajectory 16/100 | Steps: 46 | Reward: 0.54 | Success: ✓ | Total: 912 steps
Trajectory 17/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 944 steps
Trajectory 18/100 | Steps: 18 | Reward: 0.82 | Success: ✓ | Total: 962 steps
Trajectory 19/100 | Steps: 43 | Reward: 0.57 | Success: ✓ | Total: 1,005 steps
Trajectory 20/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 1,037 steps
Trajectory 21/100 | Steps: 4 | Reward: 0.96 | Success: ✓ | Total: 1,041 steps
Trajectory 22/100 | Steps: 51 | Reward: 0.49 | Success: ✓ | Total: 1,092 steps
Trajectory 23/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 1,292 steps
Trajectory 24/100 | Steps: 30 | Reward: 0.70 | Success: ✓ | Total: 1,322 steps
Trajectory 25/100 | Steps: 22 | Reward: 0.78 | Success: ✓ | Total: 1,344 steps
Trajectory 26/100 | Steps: 23 | Reward: 0.77 | Success: ✓ | Total: 1,367 steps
Trajectory 27/100 | Steps: 35 | Reward: 0.65 | Success: ✓ | Total: 1,402 steps
Trajectory 28/100 | Steps: 35 | Reward: 0.65 | Success: ✓ | Total: 1,437 steps
Trajectory 29/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 1,471 steps
Trajectory 30/100 | Steps: 16 | Reward: 0.84 | Success: ✓ | Total: 1,487 steps
Trajectory 31/100 | Steps: 36 | Reward: 0.64 | Success: ✓ | Total: 1,523 steps
Trajectory 32/100 | Steps: 21 | Reward: 0.79 | Success: ✓ | Total: 1,544 steps
Trajectory 33/100 | Steps: 118 | Reward: -0.18 | Success: ✓ | Total: 1,662 steps
Trajectory 34/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 1,862 steps
Trajectory 35/100 | Steps: 56 | Reward: 0.44 | Success: ✓ | Total: 1,918 steps
Trajectory 36/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 1,950 steps
Trajectory 37/100 | Steps: 37 | Reward: 0.63 | Success: ✓ | Total: 1,987 steps
Trajectory 38/100 | Steps: 28 | Reward: 0.72 | Success: ✓ | Total: 2,015 steps
Trajectory 39/100 | Steps: 65 | Reward: 0.35 | Success: ✓ | Total: 2,080 steps
Trajectory 40/100 | Steps: 28 | Reward: 0.72 | Success: ✓ | Total: 2,108 steps
Trajectory 41/100 | Steps: 31 | Reward: 0.69 | Success: ✓ | Total: 2,139 steps
Trajectory 42/100 | Steps: 27 | Reward: 0.73 | Success: ✓ | Total: 2,166 steps
Trajectory 43/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 2,198 steps
Trajectory 44/100 | Steps: 30 | Reward: 0.70 | Success: ✓ | Total: 2,228 steps
Trajectory 45/100 | Steps: 10 | Reward: 0.90 | Success: ✓ | Total: 2,238 steps
Trajectory 46/100 | Steps: 26 | Reward: 0.74 | Success: ✓ | Total: 2,264 steps
Trajectory 47/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 2,298 steps
Trajectory 48/100 | Steps: 41 | Reward: 0.59 | Success: ✓ | Total: 2,339 steps
Trajectory 49/100 | Steps: 29 | Reward: 0.71 | Success: ✓ | Total: 2,368 steps
Trajectory 50/100 | Steps: 22 | Reward: 0.78 | Success: ✓ | Total: 2,390 steps
Trajectory 51/100 | Steps: 26 | Reward: 0.74 | Success: ✓ | Total: 2,416 steps
Trajectory 52/100 | Steps: 30 | Reward: 0.70 | Success: ✓ | Total: 2,446 steps
Trajectory 53/100 | Steps: 23 | Reward: 0.77 | Success: ✓ | Total: 2,469 steps
Trajectory 54/100 | Steps: 16 | Reward: 0.84 | Success: ✓ | Total: 2,485 steps
Trajectory 55/100 | Steps: 25 | Reward: 0.75 | Success: ✓ | Total: 2,510 steps
Trajectory 56/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 2,544 steps
Trajectory 57/100 | Steps: 21 | Reward: 0.79 | Success: ✓ | Total: 2,565 steps
Trajectory 58/100 | Steps: 36 | Reward: 0.64 | Success: ✓ | Total: 2,601 steps
Trajectory 59/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 2,635 steps
Trajectory 60/100 | Steps: 9 | Reward: 0.91 | Success: ✓ | Total: 2,644 steps
Trajectory 61/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 2,676 steps
Trajectory 62/100 | Steps: 29 | Reward: 0.71 | Success: ✓ | Total: 2,705 steps
Trajectory 63/100 | Steps: 31 | Reward: 0.69 | Success: ✓ | Total: 2,736 steps
Trajectory 64/100 | Steps: 33 | Reward: 0.67 | Success: ✓ | Total: 2,769 steps
Trajectory 65/100 | Steps: 6 | Reward: 0.94 | Success: ✓ | Total: 2,775 steps
Trajectory 66/100 | Steps: 28 | Reward: 0.72 | Success: ✓ | Total: 2,803 steps
Trajectory 67/100 | Steps: 13 | Reward: 0.87 | Success: ✓ | Total: 2,816 steps
Trajectory 68/100 | Steps: 23 | Reward: 0.77 | Success: ✓ | Total: 2,839 steps
Trajectory 69/100 | Steps: 20 | Reward: 0.80 | Success: ✓ | Total: 2,859 steps
Trajectory 70/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 2,883 steps
Trajectory 71/100 | Steps: 32 | Reward: 0.68 | Success: ✓ | Total: 2,915 steps
Trajectory 72/100 | Steps: 5 | Reward: 0.95 | Success: ✓ | Total: 2,920 steps
Trajectory 73/100 | Steps: 19 | Reward: 0.81 | Success: ✓ | Total: 2,939 steps
Trajectory 74/100 | Steps: 20 | Reward: 0.80 | Success: ✓ | Total: 2,959 steps
Trajectory 75/100 | Steps: 141 | Reward: -0.41 | Success: ✓ | Total: 3,100 steps
Trajectory 76/100 | Steps: 19 | Reward: 0.81 | Success: ✓ | Total: 3,119 steps
Trajectory 77/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 3,319 steps
Trajectory 78/100 | Steps: 33 | Reward: 0.67 | Success: ✓ | Total: 3,352 steps
Trajectory 79/100 | Steps: 30 | Reward: 0.70 | Success: ✓ | Total: 3,382 steps
Trajectory 80/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 3,416 steps
Trajectory 81/100 | Steps: 35 | Reward: 0.65 | Success: ✓ | Total: 3,451 steps
Trajectory 82/100 | Steps: 26 | Reward: 0.74 | Success: ✓ | Total: 3,477 steps
Trajectory 83/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 3,501 steps
Trajectory 84/100 | Steps: 14 | Reward: 0.86 | Success: ✓ | Total: 3,515 steps
Trajectory 85/100 | Steps: 37 | Reward: 0.63 | Success: ✓ | Total: 3,552 steps
Trajectory 86/100 | Steps: 9 | Reward: 0.91 | Success: ✓ | Total: 3,561 steps
Trajectory 87/100 | Steps: 200 | Reward: -2.00 | Success: ✗ | Total: 3,761 steps
Trajectory 88/100 | Steps: 40 | Reward: 0.60 | Success: ✓ | Total: 3,801 steps
Trajectory 89/100 | Steps: 86 | Reward: 0.14 | Success: ✓ | Total: 3,887 steps
Trajectory 90/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 3,911 steps
Trajectory 91/100 | Steps: 13 | Reward: 0.87 | Success: ✓ | Total: 3,924 steps
Trajectory 92/100 | Steps: 23 | Reward: 0.77 | Success: ✓ | Total: 3,947 steps
Trajectory 93/100 | Steps: 35 | Reward: 0.65 | Success: ✓ | Total: 3,982 steps
Trajectory 94/100 | Steps: 19 | Reward: 0.81 | Success: ✓ | Total: 4,001 steps
Trajectory 95/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 4,025 steps
Trajectory 96/100 | Steps: 26 | Reward: 0.74 | Success: ✓ | Total: 4,051 steps
Trajectory 97/100 | Steps: 23 | Reward: 0.77 | Success: ✓ | Total: 4,074 steps
Trajectory 98/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 4,098 steps
Trajectory 99/100 | Steps: 24 | Reward: 0.76 | Success: ✓ | Total: 4,122 steps
Trajectory 100/100 | Steps: 34 | Reward: 0.66 | Success: ✓ | Total: 4,156 steps

============================================================
HDF5 File Information
============================================================
File: trajectory_data/gradients.h5
Size: 1138.5 MB

Metadata:
avg_reward: 0.5243999999999998
gradient_names: ['encoder.0.0.bias' 'encoder.0.0.weight' 'encoder.0.2.bias'
'encoder.0.2.weight' 'encoder.1.0.bias' 'encoder.1.0.weight'
'encoder.1.2.bias' 'encoder.1.2.weight' 'encoder.2.0.bias'
'encoder.2.0.weight' 'encoder.2.2.bias' 'encoder.2.2.weight' 'fc.0.bias'
'fc.0.weight' 'policy.bias' 'policy.weight' 'value.bias' 'value.weight']
gradient_shapes: ['(32,)' '(32, 3, 8, 8)' '(32,)' '(32,)' '(64,)' '(64, 32, 4, 4)' '(64,)'
'(64,)' '(64,)' '(64, 64, 3, 3)' '(64,)' '(64,)' '(512,)' '(512, 6400)'
'(5,)' '(5, 512)' '(1,)' '(1, 512)']
gradient_size: 3356646
num_trajectories: 100
success_rate: 0.91
total_steps: 4156

Datasets:
actions: shape=(4156,), dtype=int8, size=0.0 MB
done: shape=(4156,), dtype=bool, size=0.0 MB
episode_ids: shape=(4156,), dtype=int32, size=0.0 MB
gradients: shape=(4156, 3356646), dtype=float16, size=26607.9 MB
images: shape=(4156, 3, 84, 84), dtype=uint8, size=83.9 MB
rewards: shape=(4156,), dtype=float32, size=0.0 MB

Done!
