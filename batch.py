from commons import *
from scipy.io.wavfile import write
from IPython.display import Audio
import sounddevice as REC
import numpy as np
from supercollider import Server, Synth
from scipy.io import wavfile
from math import ceil, floor
import json
import librosa
from tqdm import tqdm


def sample_space(context):
    server = Server()

    # Recording properties
    SAMPLE_RATE = 44100
    SECONDS = context["SAMPLE_LENGTH"]
    # Recording device
    REC.default.device = 'Soundflower (2ch)'

    i = 0
    while i < context["PATCHES_IN_TOTAL"]:
        patch = sample_params(context)
        synth = Synth(server, "superfm", patch)

        recording = REC.rec( int(SECONDS * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 1)
        REC.wait()

        write(f"batch/audio_{context['name']}/{i}.wav", SAMPLE_RATE, recording)
        json.dump(patch, open(f"batch/patches_{context['name']}/{i}.json", "w"))

        synth.free()

        i += 1

def compute_similarity(context):
    SR = 44100

    SR, examplar = wavfile.read(f"batch/audio_{context['name']}/{0}.wav")
    patches = np.zeros((context["PATCHES_IN_TOTAL"], len(examplar)), dtype=np.float32)

    for i in tqdm(range(context["PATCHES_IN_TOTAL"])):
        _, data = wavfile.read(f"batch/audio_{context['name']}/{i}.wav") # load the data
        patches[i]=data
    patches = (patches.T - np.mean(patches, axis=1)).T # subtract mean

    examplar = librosa.feature.mfcc(y=examplar, sr=SR).flatten()
    mffcs = np.zeros((context["PATCHES_IN_TOTAL"], len(examplar)), dtype=np.float32)
    for i in tqdm(range(context["PATCHES_IN_TOTAL"])):
        mffcs[i, :] = librosa.feature.mfcc(y=patches[i], sr=SR).flatten()

    normalised = (mffcs.T / np.linalg.norm(mffcs, axis=1).T).T
    similarity = np.dot(normalised, normalised.T)
    np.fill_diagonal(similarity, 0.0)

    with open(f"batch/similarity_{context['name']}.npy", "wb") as f:
        np.save(f, similarity)

def compose(context, seeds):
    similarity = np.load(open(f"batch/similarity_{context['name']}.npy", "rb"))

    assert context["PATCHES_IN_COMPOSITION"] % len(seeds) == 0
    patches_per_seed = int(context["PATCHES_IN_COMPOSITION"] / len(seeds))

    composition = [seeds[0]]

    for seed_idx in range(len(seeds)):
        for _ in range(patches_per_seed):
            last_patch = composition[-1]
            next_patch = np.argmax(similarity[last_patch,:] + similarity[seeds[seed_idx],:])

            similarity[last_patch, next_patch] = 0.0
            similarity[next_patch, last_patch] = 0.0

            composition.append(int(next_patch))

    assert len(composition) == (context["PATCHES_IN_COMPOSITION"] + 1)
    json.dump(composition, open(f"out/composition_{context['name']}_seeds_{'_'.join([str(s) for s in seeds])}.json", "w"))

    mashed = []
    for i in range(1, len(composition), 2):
        j = i - 1

        sr, data1 = wavfile.read(f'batch/audio_{context["name"]}/{composition[j]}.wav')
        sr, data2 = wavfile.read(f'batch/audio_{context["name"]}/{composition[i]}.wav')

        if context["name"]=="flat":
            f = lambda data: fade_in_and_out(data[:int(0.15*sr)], sr, 0.001)
            data1 = f(data1); data2 = f(data2)
            mashed.append(maad_crossfader(data1, data2, sr, 0.05))

    with open(f"out/audio_{context['name']}_seed_{'_'.join([str(s) for s in seeds])}.wav", 'wb') as f:
        f.write(Audio(np.concatenate(mashed), rate=sr).data)

def maad_crossfader(s1, s2, fs, fade_len):
    fade_in = np.sqrt(np.arange(0, fs * fade_len) / (fs * fade_len))
    fade_out = np.flip(fade_in)
    fstart_idx = int(len(s1) - (fs * fade_len) - 1)
    fstop_idx = int(fs * fade_len)
    s1_fade_out = s1[fstart_idx:-1] * fade_out
    s2_fade_in = s2[0:fstop_idx] * fade_in
    s_out = np.concatenate(
        [s1[0 : fstart_idx + 1], s1_fade_out + s2_fade_in, s2[fstop_idx - 1 : -1]]
    )
    return s_out

def fade_in(audio, sr, seconds_fading=8):
    samples_fading = int(sr*seconds_fading)
    fader = np.concatenate([
        np.arange(0.0, 1.0, 1.0/samples_fading),
        np.ones(len(audio)-samples_fading),
    ])
    return audio*fader

def fade_out(audio, sr, seconds_fading=8):
    samples_fading = int(sr*seconds_fading)
    fader = np.concatenate([
        np.ones(len(audio)-samples_fading),
        np.arange(1.0, 0.0, -1.0/samples_fading),
    ])
    return audio*fader

def fade_in_and_out(audio, sr, seconds_fading=8):
    return fade_out(fade_in(audio, sr, seconds_fading=seconds_fading), sr, seconds_fading=seconds_fading)


if __name__=="__main__":
    # sample_space(flat)
    # compute_similarity(flat)
    compose(flat, seeds=[89, 4, 20, 27, 42, 44, 54, 65, 115, 89])
