import parselmouth
from parselmouth.praat import call
import math
import numpy as np
from typing import Any, Dict, List


def syllable_nuclei(fileAddress: str) -> Dict[str, Any]:
    """
    Args:
        fileAddress (str): The file path for the file we want to examine

    Returns:
        dict[str, Any]
    """
    silencedb = -25
    mindip = 2
    minpause = 0.3
    sound = parselmouth.Sound(fileAddress)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    asd = speakingtot / voicedcount
    speech_to_pause_ratio = np.nan
    if originaldur - speakingtot != 0:
      speech_to_pause_ratio = speakingtot/ (originaldur - speakingtot)
    
    speechrate_dictionary = {'speechrate(nsyll / dur)': speakingrate,
    "articulation rate(nsyll / phonationtime)": articulationrate,
    "Speech-to-pause ratio": speech_to_pause_ratio}
    
    return speechrate_dictionary


def get_final_acoustic_feat_set() -> List[str]:
    """Get the list of features we want for acoustic feature set

    Returns:
        list[str]: The list of features
    """
    finalAcousticFeatSet = ['F0semitoneFrom27.5Hz_sma3nz_amean','F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
    'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
    'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
    'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
    'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
    'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
    'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
    'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
    'jitterLocal_sma3nz_amean','jitterLocal_sma3nz_stddevNorm',
    'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm',
    'F2frequency_sma3nz_amean','F2frequency_sma3nz_stddevNorm','F3frequency_sma3nz_amean',
    'F3frequency_sma3nz_stddevNorm','VoicedSegmentsPerSec','MeanVoicedSegmentLengthSec',
    'StddevVoicedSegmentLengthSec','MeanUnvoicedSegmentLength','StddevUnvoicedSegmentLength',
    'speechrate(nsyll / dur)','articulation rate(nsyll / phonationtime)',
    'Speech-to-pause ratio']
    
    return finalAcousticFeatSet
