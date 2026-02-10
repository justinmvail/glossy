/**
 * Playback System with Observer Pattern
 *
 * Provides a decoupled playback system using EventTarget for observer pattern.
 * Components can subscribe to events instead of being tightly coupled.
 *
 * Events emitted:
 * - 'frame-added': { frame, sessionIndex, frameIndex }
 * - 'seek': { index, frame, session }
 * - 'play': {}
 * - 'pause': {}
 * - 'session-change': { sessionIndex, totalFrames }
 * - 'state-change': { playing, sessionIndex, frameIndex }
 */

class PlaybackState extends EventTarget {
  constructor(options = {}) {
    super();
    this.sessions = [];
    this.currentSession = -1;  // Index of session being recorded to
    this.viewSession = -1;     // -1 = view all, >=0 = specific session
    this.playIndex = 0;        // Current frame index in view
    this.playing = false;
    this.playTimer = null;
    this.playSpeed = options.playSpeed || 200;  // ms between frames
    this.maxSessions = options.maxSessions || 50;
    this.maxFramesPerSession = options.maxFramesPerSession || 2000;

    // Bind methods for use as callbacks
    this._tick = this._tick.bind(this);
  }

  // ==================== Recording API ====================

  /**
   * Start a new recording session with a label.
   * @param {string} label - Session label (e.g., 'Skeleton', 'Edits')
   * @returns {number} Session index
   */
  startSession(label) {
    // Enforce max sessions limit
    if (this.sessions.length >= this.maxSessions) {
      this.sessions.shift();
    }

    const session = {
      label: label || 'Session',
      frames: [],
      time: Date.now()
    };
    this.sessions.push(session);
    this.currentSession = this.sessions.length - 1;

    this._emit('session-change', {
      sessionIndex: this.currentSession,
      totalFrames: this._getTotalFrameCount()
    });

    return this.currentSession;
  }

  /**
   * Record a frame to the current session.
   * @param {object} frameData - { strokes, markers, status }
   * @returns {boolean} True if frame was recorded
   */
  recordFrame(frameData) {
    // Auto-create 'Edits' session if no active session
    if (this.currentSession < 0) {
      const lastSession = this.sessions[this.sessions.length - 1];
      if (lastSession && lastSession.label === 'Edits' &&
          lastSession.frames.length < this.maxFramesPerSession) {
        this.currentSession = this.sessions.length - 1;
      } else {
        this.startSession('Edits');
      }
    }

    const session = this.sessions[this.currentSession];
    if (!session || session.frames.length >= this.maxFramesPerSession) {
      return false;
    }

    // Deep clone the frame data
    const frame = {
      strokes: JSON.parse(JSON.stringify(frameData.strokes || [])),
      markers: JSON.parse(JSON.stringify(frameData.markers || [])),
      status: frameData.status || ''
    };

    session.frames.push(frame);
    const frameIndex = session.frames.length - 1;
    const totalFrames = this._getTotalFrameCount();

    // Update view position if not playing
    if (!this.playing) {
      if (this.viewSession === -1) {
        this.playIndex = totalFrames - 1;
      } else {
        this.viewSession = this.currentSession;
        this.playIndex = frameIndex;
      }
    }

    this._emit('frame-added', {
      frame,
      sessionIndex: this.currentSession,
      frameIndex
    });

    return true;
  }

  /**
   * Stop the current recording session.
   */
  stopSession() {
    if (this.currentSession >= 0) {
      // Update view to end of current session
      if (this.viewSession === -1) {
        this.playIndex = this._getTotalFrameCount() - 1;
      } else {
        this.viewSession = this.currentSession;
        const session = this.sessions[this.currentSession];
        this.playIndex = session ? session.frames.length - 1 : 0;
      }
    }
    this.currentSession = -1;

    this._emit('session-change', {
      sessionIndex: -1,
      totalFrames: this._getTotalFrameCount()
    });
  }

  // ==================== Playback API ====================

  /**
   * Seek to a specific frame index.
   * @param {number} index - Frame index in current view
   * @returns {object|null} Frame data or null if invalid
   */
  seek(index) {
    const frames = this._getActiveFrames();
    if (frames.length === 0) return null;

    index = Math.max(0, Math.min(index, frames.length - 1));
    this.playIndex = index;

    const entry = frames[index];

    this._emit('seek', {
      index,
      frame: entry.frame,
      session: entry.label,
      sessionIndex: entry.sessionIndex,
      localIndex: entry.localIndex,
      totalFrames: frames.length
    });

    return entry.frame;
  }

  /**
   * Start playback.
   */
  play() {
    if (this.sessions.length === 0) return;
    if (this.playing) return;

    const frames = this._getActiveFrames();
    if (frames.length === 0) return;

    // Restart from beginning if at end
    if (this.playIndex >= frames.length - 1) {
      this.playIndex = 0;
      this.seek(0);
    }

    this.playing = true;
    this.playTimer = setInterval(this._tick, this.playSpeed);

    this._emit('play', {});
    this._emit('state-change', {
      playing: true,
      sessionIndex: this.viewSession,
      frameIndex: this.playIndex
    });
  }

  /**
   * Pause playback.
   */
  pause() {
    if (!this.playing) return;

    clearInterval(this.playTimer);
    this.playTimer = null;
    this.playing = false;

    this._emit('pause', {});
    this._emit('state-change', {
      playing: false,
      sessionIndex: this.viewSession,
      frameIndex: this.playIndex
    });
  }

  /**
   * Toggle play/pause.
   * @returns {boolean} New playing state
   */
  toggle() {
    if (this.playing) {
      this.pause();
    } else {
      this.play();
    }
    return this.playing;
  }

  /**
   * Step forward or backward by a number of frames.
   * @param {number} delta - Number of frames to step (+1 or -1)
   */
  step(delta) {
    if (this.playing) this.pause();
    this.seek(this.playIndex + delta);
  }

  /**
   * Jump to first frame.
   */
  first() {
    if (this.playing) this.pause();
    this.seek(0);
  }

  /**
   * Jump to last frame.
   */
  last() {
    if (this.playing) this.pause();
    const frames = this._getActiveFrames();
    this.seek(frames.length - 1);
  }

  /**
   * Set playback speed.
   * @param {number} ms - Milliseconds between frames
   */
  setSpeed(ms) {
    this.playSpeed = ms;
    if (this.playing) {
      clearInterval(this.playTimer);
      this.playTimer = setInterval(this._tick, this.playSpeed);
    }
  }

  /**
   * Switch to viewing a specific session or all sessions.
   * @param {number} sessionIndex - Session index, or -1 for all
   * @param {boolean} resetPosition - If true, reset to frame 0. Default: true
   */
  selectSession(sessionIndex, resetPosition = true) {
    this.viewSession = sessionIndex;

    if (resetPosition) {
      this.playIndex = 0;
      this.seek(0);
    } else {
      // Clamp playIndex to valid range for new view
      const frames = this._getActiveFrames();
      this.playIndex = Math.min(this.playIndex, Math.max(0, frames.length - 1));
    }

    this._emit('session-change', {
      sessionIndex,
      totalFrames: this._getActiveFrames().length
    });
  }

  // ==================== Query API ====================

  /**
   * Get all frames across all sessions (flat list).
   * @returns {Array} Array of { frame, sessionIndex, label, localIndex }
   */
  getAllFrames() {
    return this._getAllFrames();
  }

  /**
   * Get frames for current view (filtered by viewSession).
   * @returns {Array} Array of { frame, sessionIndex, label, localIndex }
   */
  getActiveFrames() {
    return this._getActiveFrames();
  }

  /**
   * Get current frame.
   * @returns {object|null} Current frame or null
   */
  getCurrentFrame() {
    const frames = this._getActiveFrames();
    if (frames.length === 0 || this.playIndex >= frames.length) return null;
    return frames[this.playIndex].frame;
  }

  /**
   * Get session list for UI.
   * @returns {Array} Array of { label, frameCount, index }
   */
  getSessionList() {
    return this.sessions.map((s, i) => ({
      label: s.label,
      frameCount: s.frames.length,
      index: i
    }));
  }

  /**
   * Get current state snapshot.
   * @returns {object} Current state
   */
  getState() {
    const frames = this._getActiveFrames();
    return {
      playing: this.playing,
      playIndex: this.playIndex,
      viewSession: this.viewSession,
      currentSession: this.currentSession,
      totalFrames: frames.length,
      sessionCount: this.sessions.length,
      playSpeed: this.playSpeed
    };
  }

  /**
   * Check if there are any recorded frames.
   * @returns {boolean}
   */
  hasFrames() {
    return this._getTotalFrameCount() > 0;
  }

  // ==================== Persistence API ====================

  /**
   * Export all sessions as JSON for localStorage.
   * @returns {string} JSON string
   */
  export() {
    return JSON.stringify({
      sessions: this.sessions,
      viewSession: this.viewSession,
      playIndex: this.playIndex
    });
  }

  /**
   * Import sessions from JSON.
   * @param {string} json - JSON string from export()
   */
  import(json) {
    try {
      const data = JSON.parse(json);
      this.sessions = data.sessions || [];
      this.viewSession = data.viewSession ?? -1;
      this.playIndex = data.playIndex ?? 0;
      this.currentSession = -1;

      this._emit('session-change', {
        sessionIndex: this.viewSession,
        totalFrames: this._getTotalFrameCount()
      });

      // Seek to restore position
      if (this._getActiveFrames().length > 0) {
        this.seek(Math.min(this.playIndex, this._getActiveFrames().length - 1));
      }
    } catch (e) {
      console.error('Failed to import playback state:', e);
    }
  }

  /**
   * Clear all sessions.
   */
  clear() {
    if (this.playing) this.pause();
    this.sessions = [];
    this.currentSession = -1;
    this.viewSession = -1;
    this.playIndex = 0;

    this._emit('session-change', {
      sessionIndex: -1,
      totalFrames: 0
    });
  }

  // ==================== Internal Methods ====================

  _tick() {
    const frames = this._getActiveFrames();
    if (this.playIndex >= frames.length - 1) {
      this.pause();
      return;
    }
    this.seek(this.playIndex + 1);
  }

  _getAllFrames() {
    const all = [];
    for (let si = 0; si < this.sessions.length; si++) {
      const session = this.sessions[si];
      for (let fi = 0; fi < session.frames.length; fi++) {
        all.push({
          frame: session.frames[fi],
          sessionIndex: si,
          label: session.label,
          localIndex: fi
        });
      }
    }
    return all;
  }

  _getActiveFrames() {
    if (this.viewSession === -1) {
      return this._getAllFrames();
    }
    if (this.viewSession >= 0 && this.viewSession < this.sessions.length) {
      const session = this.sessions[this.viewSession];
      return session.frames.map((f, fi) => ({
        frame: f,
        sessionIndex: this.viewSession,
        label: session.label,
        localIndex: fi
      }));
    }
    return [];
  }

  _getTotalFrameCount() {
    return this.sessions.reduce((sum, s) => sum + s.frames.length, 0);
  }

  _emit(eventName, detail) {
    this.dispatchEvent(new CustomEvent(eventName, { detail }));
  }
}

// ==================== UI Binding Helpers ====================

/**
 * Create a UI controller that binds PlaybackState to DOM elements.
 *
 * @param {PlaybackState} playback - The playback state instance
 * @param {object} elements - DOM element references
 * @param {function} onSeek - Callback when seeking (receives frame data)
 * @returns {object} Controller with update methods
 */
function createPlaybackUI(playback, elements, onSeek) {
  const {
    panel,           // Container panel (shown/hidden based on frames)
    sessionSelect,   // <select> for session dropdown
    timeline,        // <input type="range"> for seeking
    frameInfo,       // Element to show step description
    frameNum,        // Element to show frame number (optional)
    playPauseBtn,    // Play/Pause button
    speedSelect      // <select> for speed
  } = elements;

  function updatePanel() {
    const state = playback.getState();
    if (panel) {
      panel.style.display = state.sessionCount > 0 ? 'block' : 'none';
    }
  }

  function updateSessionSelect() {
    if (!sessionSelect) return;
    const sessions = playback.getSessionList();
    const totalFrames = playback.getAllFrames().length;

    let html = `<option value="-1">All (${totalFrames}f)</option>`;
    html += sessions.map(s =>
      `<option value="${s.index}">${s.label} (${s.frameCount}f)</option>`
    ).join('');

    sessionSelect.innerHTML = html;
    sessionSelect.value = playback.viewSession;
  }

  function updateTimeline() {
    if (!timeline) return;
    const frames = playback.getActiveFrames();
    timeline.max = Math.max(0, frames.length - 1);
    timeline.value = Math.min(playback.playIndex, frames.length - 1);
  }

  function updateFrameInfo() {
    const frames = playback.getActiveFrames();
    const idx = playback.playIndex;

    // Update frame number display
    if (frameNum) {
      frameNum.textContent = frames.length > 0 ? `F${idx + 1}/${frames.length}` : 'F0/0';
    }

    // Update step description
    if (frameInfo) {
      if (frames.length > 0 && idx < frames.length) {
        const entry = frames[idx];
        const status = entry.frame.status || '';
        frameInfo.textContent = status;
        frameInfo.title = status;  // Tooltip for full text
      } else {
        frameInfo.textContent = '';
        frameInfo.title = '';
      }
    }
  }

  function updatePlayButton() {
    if (!playPauseBtn) return;
    playPauseBtn.textContent = playback.playing ? 'Pause' : 'Play';
  }

  function updateAll() {
    updatePanel();
    updateSessionSelect();
    updateTimeline();
    updateFrameInfo();
    updatePlayButton();
  }

  // Subscribe to events
  playback.addEventListener('frame-added', updateAll);
  playback.addEventListener('session-change', updateAll);
  playback.addEventListener('seek', (e) => {
    updateTimeline();
    updateFrameInfo();
    if (onSeek) onSeek(e.detail);
  });
  playback.addEventListener('play', updatePlayButton);
  playback.addEventListener('pause', updatePlayButton);

  // Bind UI controls
  if (sessionSelect) {
    sessionSelect.addEventListener('change', () => {
      playback.selectSession(parseInt(sessionSelect.value, 10));
    });
  }

  if (timeline) {
    timeline.addEventListener('input', () => {
      playback.seek(parseInt(timeline.value, 10));
    });
  }

  if (speedSelect) {
    speedSelect.addEventListener('change', () => {
      playback.setSpeed(parseInt(speedSelect.value, 10));
    });
  }

  // Initial update
  updateAll();

  return {
    update: updateAll,
    updatePanel,
    updateSessionSelect,
    updateTimeline,
    updateFrameInfo,
    updatePlayButton
  };
}

// Export for module systems (if available) or attach to window
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { PlaybackState, createPlaybackUI };
} else if (typeof window !== 'undefined') {
  window.PlaybackState = PlaybackState;
  window.createPlaybackUI = createPlaybackUI;
}
