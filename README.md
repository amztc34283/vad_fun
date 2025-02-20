### My approach/strategy
I first did a lot of research on VAD in general and created a new algorithm (exponential moving average) to detect speech or noise. I also used the “state-of-the-art” VAD algorithm (webrtcvad) to compare my result with them making sure my algorithm is on-par. Then I started to work on the real task, detecting pause vs ended speech, by analyzing the data to extract useful insights. I tried a lot of things and ended up using the pitch to detect filler words so I could flag the system for potential pause if a filler word was within the window of interest. By reusing my algorithm, I could already detect filler words, speech, and noise, and they are basically all we need. Nonetheless, this information will be stored in a queue for further processing.

### Key features/decisions used
There are 3 states defined in the Moore Machine of the system: ENDED, SPEAKING, PAUSED. The initial state is always at ENDED. The most difficult part of the logic is at SPEAKING state determining if the speech has ended or paused. Since we have kept information about each chunk of the past, we were able to use them to our advantage.

ENDED:
If speech is detected, then we transition to SPEAKING 

SPEAKING:
If there were no speech or filler words for some time X, then we transition to ENDED.
If there were a filler word at some Y time before and had been silenced for some time Z, then we transition to PAUSED.

PAUSED:
If speech is detected, then we transition to SPEAKING.
If no speech is detected for some time W, then we transition to ENDED.

### Challenges faced
I had to perform a lot of different analyses on the data to make sure I could figure out an angle of attack to each sub-problem. In addition, there are some parameters I had to tune in order to make it work such as time, window size, leniency for detecting filler words, etc. but they were also the fun of the projects. Last but not least, there were bugs like all programs in the world but I was able to fix them.

### Possible improvements with more time
I think the solution can run in real-time already as all calculations are very fast and can be done within a short amount of time; however, I might consider running streaming on the pitch finding library to extract the last bit of performance. In general, I think this problem is well-suited to using deep learning because of the huge varieties of languages in the world. However, you might run into some performance problems. Eventually, I want to make it more robust, maybe using facial expression, demographic information to dynamically adjust the parameters of the system.


