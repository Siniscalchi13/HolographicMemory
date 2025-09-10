/**
 * HolographicMemory 3D Wave Visualization
 * Adapted from SmartHaus website implementation
 * Uses global THREE object loaded from CDN
 */

// Check if THREE is available globally (loaded via script tag)
function getTHREE() {
  if (typeof THREE !== 'undefined') {
    return Promise.resolve(THREE);
  }

  // If not loaded, try to load it dynamically
  if (!document.querySelector('script[src*="three"]')) {
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js';
    script.onload = () => console.log('[3D] Three.js loaded successfully');
    script.onerror = () => {
      console.error('[3D] Failed to load Three.js - WebGL may not be supported');
      throw new Error('Three.js library not available - WebGL may not be supported on this device');
    };
    document.head.appendChild(script);
  }

  // Wait for THREE to be available with timeout
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const maxAttempts = 100; // 10 seconds

    const checkTHREE = () => {
      attempts++;
      if (typeof THREE !== 'undefined') {
        resolve(THREE);
      } else if (attempts < maxAttempts) {
        setTimeout(checkTHREE, 100);
      } else {
        reject(new Error('Three.js load timeout - WebGL may not be supported on this device'));
      }
    };
    setTimeout(checkTHREE, 100);
  });
}

// Check WebGL support
function checkWebGLSupport() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && canvas.getContext('webgl'));
  } catch (e) {
    return false;
  }
}

// Load OrbitControls
function loadOrbitControls(THREE) {
  return new Promise((resolve) => {
    if (THREE.OrbitControls) {
      console.log('[3D] OrbitControls already available');
      resolve(THREE.OrbitControls);
      return;
    }

    // Try to load OrbitControls from CDN
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js';
    script.onload = () => {
      console.log('[3D] OrbitControls loaded successfully');
      resolve(THREE.OrbitControls);
    };
    script.onerror = () => {
      console.warn('[3D] OrbitControls CDN failed, trying alternative');

      // Try alternative CDN
      const altScript = document.createElement('script');
      altScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/examples/js/controls/OrbitControls.js';
      altScript.onload = () => {
        console.log('[3D] OrbitControls loaded from alternative CDN');
        resolve(THREE.OrbitControls);
      };
      altScript.onerror = () => {
        console.warn('[3D] All OrbitControls CDNs failed, using basic controls');
        resolve(null);
      };
      document.head.appendChild(altScript);
    };
    document.head.appendChild(script);

    // Timeout after 8 seconds
    setTimeout(() => {
      if (!THREE.OrbitControls) {
        console.warn('[3D] OrbitControls load timeout, using basic controls');
        resolve(null);
      }
    }, 8000);
  });
}

class WaveViewer3D {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.container = null;
    this.animationId = null;
    this.controls = null;
    
    // Wave data
    this.amplitude = [];
    this.phase = [];
    this.rings = [];
    this.mode = 'rings'; // 'rings' | 'surface'
    
    // Animation state
    this.time = 0;
    this.isAnimating = true;
    
    // Performance
    this.frameCount = 0;
    this.lastFpsUpdate = 0;
    this.fps = 60;
  }

  async initScene(container) {
    this.container = container;

    // Check WebGL support first
    if (!checkWebGLSupport()) {
      throw new Error('WebGL not supported on this device');
    }

    // Get THREE object
    const THREE = await getTHREE();

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);
    
    // Camera
    this.camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 3, 10);
    
    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    // Load OrbitControls
    const OrbitControls = await loadOrbitControls(THREE);

    // Controls
    if (OrbitControls) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enablePan = true;
      this.controls.enableZoom = true;
      this.controls.enableRotate = true;
      this.controls.autoRotate = false;
      this.controls.minDistance = 5;
      this.controls.maxDistance = 20;
      console.log('[3D] OrbitControls initialized successfully');
    } else {
      console.warn('[3D] OrbitControls not available, using basic mouse controls');
      // Add basic mouse controls as fallback
      this.setupBasicControls();
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);
    
    const pointLight1 = new THREE.PointLight(0xffffff, 1);
    pointLight1.position.set(10, 10, 10);
    this.scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0xff00ff, 0.5);
    pointLight2.position.set(-10, -10, -10);
    this.scene.add(pointLight2);
    
    const pointLight3 = new THREE.PointLight(0x00ffff, 0.5);
    pointLight3.position.set(10, -10, 10);
    this.scene.add(pointLight3);
    
    // Controls already initialized above
    
    // Add reference beams
    this.addReferenceBeams();
    
    // Start animation
    this.animate();
    
    // Handle resize
    window.addEventListener('resize', () => this.onWindowResize());
    
    return this;
  }
  
  
  addReferenceBeams() {
    // Reference beam (cyan)
    const refGeometry = new THREE.CylinderGeometry(0.05, 0.05, 6);
    const refMaterial = new THREE.MeshStandardMaterial({
      color: 0x00ffff,
      emissive: 0x00ffff,
      emissiveIntensity: 1,
      opacity: 0.8,
      transparent: true
    });
    const refBeam = new THREE.Mesh(refGeometry, refMaterial);
    refBeam.position.set(-2, 0, 0);
    refBeam.rotation.z = Math.PI / 2;
    this.scene.add(refBeam);
    
    // Object beam (magenta)
    const objGeometry = new THREE.CylinderGeometry(0.05, 0.05, 6);
    const objMaterial = new THREE.MeshStandardMaterial({
      color: 0xff00ff,
      emissive: 0xff00ff,
      emissiveIntensity: 1,
      opacity: 0.8,
      transparent: true
    });
    const objBeam = new THREE.Mesh(objGeometry, objMaterial);
    objBeam.position.set(0, 0, -2);
    objBeam.rotation.x = Math.PI / 2;
    this.scene.add(objBeam);
    
    // Central hologram core
    const coreGeometry = new THREE.SphereGeometry(0.3, 32, 32);
    const coreMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.5,
      opacity: 0.5,
      transparent: true
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    this.scene.add(core);
  }
  
  updateData(amplitude, phase = null) {
    console.log(`[3D] updateData called with:`, amplitude?.length, 'amplitudes,', phase?.length, 'phases');
    console.log(`[3D] Sample amplitude data:`, amplitude?.slice(0, 5));
    console.log(`[3D] Sample phase data:`, phase?.slice(0, 5));

    this.amplitude = amplitude || [];
    this.phase = phase || this.amplitude.map((_, i) => (i / this.amplitude.length) * Math.PI * 2);

    // Clear existing rings
    this.rings.forEach(ring => this.scene.remove(ring));
    this.rings = [];

    if (this.mode === 'rings') {
      this.createRings();
    }

    console.log(`[3D] Updated with ${this.amplitude.length} samples`);
  }
  
  createRings() {
    const numRings = Math.min(10, Math.floor(this.amplitude.length / 8));
    const samplesPerRing = Math.floor(this.amplitude.length / numRings);
    
    for (let i = 0; i < numRings; i++) {
      const startIdx = i * samplesPerRing;
      const endIdx = Math.min(startIdx + samplesPerRing, this.amplitude.length);
      
      // Average amplitude and phase for this ring
      let avgAmplitude = 0;
      let avgPhase = 0;
      for (let j = startIdx; j < endIdx; j++) {
        avgAmplitude += this.amplitude[j];
        avgPhase += this.phase[j];
      }
      avgAmplitude /= (endIdx - startIdx);
      avgPhase /= (endIdx - startIdx);
      
      // Create ring geometry
      const innerRadius = 0.5 + i * 0.3;
      const outerRadius = innerRadius + 0.25;
      const geometry = new THREE.RingGeometry(innerRadius, outerRadius, 64, 8);
      
      // Map phase to HSL hue (0-360 degrees)
      const hue = ((avgPhase + Math.PI) / (2 * Math.PI)) * 360;
      const color = new THREE.Color().setHSL(hue / 360, 1, 0.5);
      
      const material = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.3 * avgAmplitude,
        opacity: 0.3 + avgAmplitude * 0.4,
        transparent: true,
        side: THREE.DoubleSide
      });
      
      const ring = new THREE.Mesh(geometry, material);
      ring.position.y = (i - numRings / 2) * 0.3;
      ring.rotation.y = (i * Math.PI) / 10;
      ring.userData = { 
        originalY: ring.position.y,
        originalRotation: ring.rotation.y,
        amplitude: avgAmplitude,
        phase: avgPhase,
        ringIndex: i
      };
      
      this.scene.add(ring);
      this.rings.push(ring);
    }
    
    // Add spiral data points in interference pattern
    this.addDataPoints();
  }
  
  addDataPoints() {
    // Remove existing data points
    const existingPoints = this.scene.children.filter(child => child.userData.isDataPoint);
    existingPoints.forEach(point => this.scene.remove(point));
    
    // Add new data points based on amplitude
    const numPoints = Math.min(100, this.amplitude.length);
    const step = Math.floor(this.amplitude.length / numPoints);
    
    for (let i = 0; i < numPoints; i++) {
      const dataIndex = i * step;
      const amp = this.amplitude[dataIndex] || 0;
      const phase = this.phase[dataIndex] || 0;
      
      // Spiral positioning
      const theta = (i / numPoints) * Math.PI * 2 * 5;
      const r = (i / numPoints) * 2.5;
      const y = (i / numPoints - 0.5) * 3;
      
      const geometry = new THREE.SphereGeometry(0.02 * (1 + amp), 6, 6);
      const material = new THREE.MeshStandardMaterial({
        color: 0xffd700,
        emissive: 0xffd700,
        emissiveIntensity: 0.5 * amp,
        opacity: 0.9,
        transparent: true
      });
      
      const point = new THREE.Mesh(geometry, material);
      point.position.set(
        Math.cos(theta) * r,
        y,
        Math.sin(theta) * r
      );
      point.userData = { 
        isDataPoint: true, 
        amplitude: amp, 
        phase: phase,
        originalPosition: point.position.clone()
      };
      
      this.scene.add(point);
    }
  }
  
  setMode(mode) {
    this.mode = mode;
    if (this.amplitude.length > 0) {
      this.updateData(this.amplitude, this.phase);
    }
  }
  
  animate() {
    this.animationId = requestAnimationFrame(() => this.animate());
    
    if (!this.isAnimating) return;
    
    this.time += 0.016; // ~60fps
    
    // Update FPS counter
    this.frameCount++;
    if (this.time - this.lastFpsUpdate > 1) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsUpdate = this.time;
    }
    
    // Animate rings
    this.rings.forEach((ring, i) => {
      if (ring.userData) {
        // Rotate based on phase and time
        ring.rotation.z = this.time * 0.1 * (i % 2 === 0 ? 1 : -1);
        
        // Pulse opacity based on amplitude
        const baseMaterial = ring.material;
        if (baseMaterial && 'opacity' in baseMaterial) {
          baseMaterial.opacity = 0.2 + Math.sin(this.time * 2 + i) * 0.1 * ring.userData.amplitude;
        }
      }
    });
    
    // Animate data points
    const dataPoints = this.scene.children.filter(child => child.userData.isDataPoint);
    dataPoints.forEach((point, i) => {
      if (point.userData && point.userData.originalPosition) {
        // Slight oscillation based on phase
        const oscillation = Math.sin(this.time * 3 + point.userData.phase) * 0.1;
        point.position.copy(point.userData.originalPosition);
        point.position.y += oscillation;
      }
    });
    
    // Update controls
    if (this.controls && typeof this.controls.update === 'function') {
      this.controls.update();
    }
    
    // Render
    this.renderer.render(this.scene, this.camera);
  }
  
  onWindowResize() {
    if (!this.container || !this.camera || !this.renderer) return;
    
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }
  
  toggleAnimation() {
    this.isAnimating = !this.isAnimating;
    return this.isAnimating;
  }
  
  getStats() {
    return {
      fps: this.fps,
      rings: this.rings.length,
      samples: this.amplitude.length,
      mode: this.mode,
      animating: this.isAnimating
    };
  }
  
  setupBasicControls() {
    this.mouse = { x: 0, y: 0, down: false };
    this.rotation = { x: 0, y: 0 };
    this.distance = 10;

    const canvas = this.renderer.domElement;

    canvas.addEventListener('mousedown', (e) => {
      this.mouse.down = true;
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!this.mouse.down) return;

      const deltaX = e.clientX - this.mouse.x;
      const deltaY = e.clientY - this.mouse.y;

      this.rotation.y += deltaX * 0.01;
      this.rotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.rotation.x + deltaY * 0.01));

      this.camera.position.x = Math.sin(this.rotation.y) * this.distance * Math.cos(this.rotation.x);
      this.camera.position.z = Math.cos(this.rotation.y) * this.distance * Math.cos(this.rotation.x);
      this.camera.position.y = Math.sin(this.rotation.x) * this.distance;

      this.camera.lookAt(0, 0, 0);

      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });

    canvas.addEventListener('mouseup', () => {
      this.mouse.down = false;
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.max(5, Math.min(20, this.distance + e.deltaY * 0.01));

      this.camera.position.x = Math.sin(this.rotation.y) * this.distance * Math.cos(this.rotation.x);
      this.camera.position.z = Math.cos(this.rotation.y) * this.distance * Math.cos(this.rotation.x);
      this.camera.position.y = Math.sin(this.rotation.x) * this.distance;

      this.camera.lookAt(0, 0, 0);
    });

    console.log('[3D] Basic mouse controls initialized');
  }

  dispose() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement && this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }
    
    // Clean up geometries and materials
    this.scene?.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose());
        } else {
          object.material.dispose();
        }
      }
    });
    
    window.removeEventListener('resize', this.onWindowResize);
  }
}

// Export for ES module usage
export { WaveViewer3D };

// Also expose globally for non-module usage
window.WaveViewer3D = WaveViewer3D;
