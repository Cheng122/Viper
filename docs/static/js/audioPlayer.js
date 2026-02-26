const audio = document.getElementById('audio-file');
const playButton = document.getElementById('audio-control-button');
const playIcon = document.getElementById('play-icon');
const playText = document.getElementById('play-text');
const audioControls = document.getElementById('audio-controls');
const seekBar = document.getElementById('seek-bar');
const timeRemaining = document.getElementById('time-remaining');
const durationDisplay = document.getElementById('duration');
const toggleVisibilityIcon = document.getElementById('toggle-visibility-icon');
const visibilityIcon = document.getElementById('visibility-icon');

let isFirstPlay = true;
let controlsVisible = true;  // The controls are initially hidden but the state should track visibility.

function toggleAudio() {
  if (audio.paused) {
    audio.play();
    playIcon.classList.remove('fa-play');
    playIcon.classList.add('fa-pause');
    playText.textContent = 'Podcast';

    if (isFirstPlay) {
      showControls();  // Show the controls when playing for the first time
      toggleVisibilityIcon.style.display = 'block';  // Show the toggle icon after the first play
      isFirstPlay = false;
    }

  } else {
    audio.pause();
    playIcon.classList.remove('fa-pause');
    playIcon.classList.add('fa-play');
    playText.textContent = 'Podcast';
  }
}

// Show controls function
function showControls() {
  audioControls.style.display = 'block';
  visibilityIcon.classList.remove('fa-eye-slash');
  visibilityIcon.classList.add('fa-eye');
  controlsVisible = true;
}

// Hide controls function
function hideControls() {
  audioControls.style.display = 'none';
  visibilityIcon.classList.remove('fa-eye');
  visibilityIcon.classList.add('fa-eye-slash');
  controlsVisible = false;
}

// Toggle controls using the eye icon
function toggleControls() {
  if (controlsVisible) {
    hideControls();
  } else {
    showControls();
  }
}

// Update the seek bar as the audio plays
audio.addEventListener('timeupdate', function () {
  const currentTime = audio.currentTime;
  const duration = audio.duration;
  seekBar.max = duration;
  seekBar.value = currentTime;

  const remaining = Math.floor(duration - currentTime);
  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;

  timeRemaining.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
});

// Allow users to seek through the audio
seekBar.addEventListener('input', function () {
  audio.currentTime = seekBar.value;
});

// Display duration when audio metadata is loaded
audio.addEventListener('loadedmetadata', function () {
  const duration = Math.floor(audio.duration);
  const minutes = Math.floor(duration / 60);
  const seconds = duration % 60;
  durationDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
});

