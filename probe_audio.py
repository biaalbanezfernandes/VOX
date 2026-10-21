import ctypes
import ctypes.wintypes

winmm = ctypes.WinDLL("winmm.dll")

class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag",      ctypes.wintypes.WORD),
        ("nChannels",       ctypes.wintypes.WORD),
        ("nSamplesPerSec",  ctypes.wintypes.DWORD),
        ("nAvgBytesPerSec", ctypes.wintypes.DWORD),
        ("nBlockAlign",     ctypes.wintypes.WORD),
        ("wBitsPerSample",  ctypes.wintypes.WORD),
        ("cbSize",          ctypes.wintypes.WORD),
    ]

WAVE_MAPPER = 0xFFFFFFFF
WAVE_FORMAT_PCM = 1
CALLBACK_NULL = 0x00000000

print("Testando waveIn API com WAVE_MAPPER (dispositivo padrao do Windows):")
for sr in [44100, 48000, 16000, 22050, 8000]:
    wfx = WAVEFORMATEX(
        wFormatTag=WAVE_FORMAT_PCM, nChannels=1,
        nSamplesPerSec=sr, nAvgBytesPerSec=sr * 2,
        nBlockAlign=2, wBitsPerSample=16, cbSize=0,
    )
    hwi = ctypes.wintypes.HANDLE()
    ret = winmm.waveInOpen(
        ctypes.byref(hwi), WAVE_MAPPER,
        ctypes.byref(wfx), 0, 0, CALLBACK_NULL,
    )
    if ret == 0:
        winmm.waveInClose(hwi)
        print(f"  {sr} Hz -> OK")
    else:
        print(f"  {sr} Hz -> FAIL (code {ret})")
