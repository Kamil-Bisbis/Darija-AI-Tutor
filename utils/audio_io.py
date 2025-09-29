import collections
import numpy as np
import sounddevice as sd
import webrtcvad

# Audio settings
SAMPLE_RATE = 16000
BLOCK_MS = 30  # 10, 20, or 30 for webrtcvad
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
MAX_RECORD_SECONDS = 8
SILENCE_TAIL_BLOCKS = 15  # about 0.45 s if BLOCK_MS = 30


def record_until_silence() -> np.ndarray:
	"""Record audio until a trailing window of silence is detected or max length reached.

	Returns:
		np.ndarray: Mono float32 PCM audio in range [-1, 1].
	"""
	vad = webrtcvad.Vad(2)  # 0-3, 2 is medium-aggressive
	stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
	frames: list[bytes] = []
	ring = collections.deque(maxlen=SILENCE_TAIL_BLOCKS)
	voiced_any = False
	try:
		stream.start()
		total_blocks = int(MAX_RECORD_SECONDS * 1000 / BLOCK_MS)
		for _ in range(total_blocks):
			block = stream.read(BLOCK_SAMPLES)[0].tobytes()
			is_voiced = vad.is_speech(block, SAMPLE_RATE)
			ring.append(is_voiced)
			frames.append(block)
			if is_voiced:
				voiced_any = True
			# if we have seen voice and now a tail of silence, stop
			if voiced_any and len(ring) == ring.maxlen and not any(ring):
				break
	finally:
		stream.stop()
		stream.close()

	# bytes to int16 numpy then to float32 [-1, 1]
	audio_bytes = b"".join(frames)
	audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
	return audio_np