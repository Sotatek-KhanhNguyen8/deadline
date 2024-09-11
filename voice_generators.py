from role_inference import EllenVoiceGenerator, TrumpVoiceGenerator, LxVoiceGenerator

role_generators = {
    f'{EllenVoiceGenerator.speaker_id}': EllenVoiceGenerator(),
    f'{TrumpVoiceGenerator.speaker_id}': TrumpVoiceGenerator(),
    f'{LxVoiceGenerator.speaker_id}': LxVoiceGenerator(),
}
