# Análise visual do vídeo "ColdFlowTest.mp4" para procurar pulsos no escoamento.
# O código:
# - lê o vídeo em /mnt/data/ColdFlowTest.mp4
# - gera uma imagem com o primeiro frame e a ROI automática detectada (baseada em variação temporal / brilho)
# - calcula a intensidade média de brilho na ROI por frame (proxy para 'massa ejetada' visível)
# - plota a série temporal da intensidade e o seu espectro (FFT) para procurar periodicidade/pulsos
# - salva os gráficos no /mnt/data para download
#
# Aviso: esta é uma análise visual/heurística — sinais visuais podem não corresponder exactamente a variações de caudal mássico.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
video_path = Path('Video/ColdFlowTest.mp4')
assert video_path.exists(), "Ficheiro de vídeo não encontrado em /mnt/data/ColdFlowTest.mp4"

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read all frames (grayscale) but downsample if very long to limit time/memory
max_frames = 3000
step = 1
if frame_count > max_frames:
    step = int(np.ceil(frame_count / max_frames))

frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % step == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    idx += 1
cap.release()

frames = np.array(frames)  # shape (N,h,w)
N = frames.shape[0]
times = np.arange(N) / fps * step

# Compute median background and difference to detect moving plume region
median_bg = np.median(frames, axis=0).astype(np.uint8)
diff = np.abs(frames.astype(int) - median_bg.astype(int)).astype(np.uint8)
# Sum differences over time to find region with most motion
motion_map = np.sum(diff, axis=0)
# Threshold motion_map to get ROI
th = np.percentile(motion_map, 90)
roi_mask = (motion_map > th).astype(np.uint8)
# Expand ROI a bit with dilation
kernel = np.ones((11,11), np.uint8)
roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
# Find bounding box of ROI
ys, xs = np.where(roi_mask)
if len(xs) == 0:
    # fallback: use center region
    cx, cy = w//2, h//2
    bw, bh = w//3, h//3
    x0, y0, x1, y1 = max(0,cx-bw//2), max(0,cy-bh//2), min(w,cx+bw//2), min(h,cy+bh//2)
else:
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # pad
    pad = 10
    x0, x1 = max(0, x0-pad), min(w, x1+pad)
    y0, y1 = max(0, y0-pad), min(h, y1+pad)

# Compute mean brightness in ROI for each frame (proxy for visible mass flow)
means = []
for i in range(N):
    roi = frames[i, y0:y1, x0:x1]
    means.append(np.mean(roi))
means = np.array(means)

# Detrend and compute FFT for periodicity detection
detrended = means - np.mean(means)
# zero pad to next power of two
nfft = 1 << (int(np.ceil(np.log2(len(detrended)))))
fft = np.fft.rfft(detrended, n=nfft)
freqs = np.fft.rfftfreq(nfft, d=1.0/(fps/step))
power = np.abs(fft)**2

# Save diagnostic plots
# 1) First frame with ROI box
first_frame_rgb = cv2.cvtColor(cv2.imread(str(video_path)) if False else cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
# But we need to ensure correct resizing: use frames[0] grayscale converted
first_frame = frames[0]
plt.figure(figsize=(6,6))
plt.imshow(first_frame, cmap='gray', vmin=0, vmax=255)
plt.title('First frame (grayscale) with detected ROI')
plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, linewidth=2))
plt.axis('off')
plt.savefig('frame_with_roi.png', bbox_inches='tight', dpi=150)
plt.close()

# 2) Time series plot
plt.figure(figsize=(8,3.5))
plt.plot(times, means)
plt.xlabel('Time (s)')
plt.ylabel('Mean intensity in ROI (arb. units)')
plt.title('Intensity in ROI vs time (proxy for visible flow)')
plt.grid(True)
plt.savefig('intensity_time_series.png', bbox_inches='tight', dpi=150)
plt.close()

# 3) FFT / Power spectrum plot (freq up to Nyquist)
plt.figure(figsize=(8,3.5))
plt.semilogy(freqs, power, linewidth=1)
plt.xlim(0, min(50, freqs.max()))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (arb. units)')
plt.title('Power spectrum of intensity signal (looking for periodic pulses)')
plt.grid(True)
plt.savefig('power_spectrum.png', bbox_inches='tight', dpi=150)
plt.close()

# 4) Save a small CSV with times and means for inspection
import pandas as pd
df = pd.DataFrame({'time_s': times, 'mean_intensity': means})
df.to_csv('intensity_timeseries.csv', index=False)

# Display results to user
print(f"Frames analyzed: {N}, FPS (video): {fps}, effective frame rate used: {fps/step:.2f} fps")
print(f"ROI bbox: x0={x0}, y0={y0}, x1={x1}, y1={y1}. Output files in\n")
# Show images inline
from IPython.display import Image, display
display(Image(str('frame_with_roi.png')))
display(Image(str('intensity_time_series.png')))
display(Image(str(-'power_spectrum.png')))

# Save df to /mnt/data for download and also show first rows
df.head(20)
