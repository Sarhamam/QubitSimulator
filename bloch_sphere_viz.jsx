import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';

// Complex number operations
const Complex = {
  mul: (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }),
  add: (a, b) => ({ re: a.re + b.re, im: a.im + b.im }),
  scale: (a, s) => ({ re: a.re * s, im: a.im * s }),
  conj: (a) => ({ re: a.re, im: -a.im }),
  abs2: (a) => a.re * a.re + a.im * a.im,
  exp: (theta) => ({ re: Math.cos(theta), im: Math.sin(theta) }),
};

// Convert state vector to Bloch coordinates
const stateToBloch = (alpha, beta) => {
  // |ψ⟩ = α|0⟩ + β|1⟩
  // x = 2 Re(α*β), y = 2 Im(α*β), z = |α|² - |β|²
  const alphaConj = Complex.conj(alpha);
  const prod = Complex.mul(alphaConj, beta);
  return {
    x: 2 * prod.re,
    y: 2 * prod.im,
    z: Complex.abs2(alpha) - Complex.abs2(beta),
  };
};

// Apply rotation gate
const applyRotation = (alpha, beta, axis, angle) => {
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);
  
  if (axis === 'x') {
    return {
      alpha: { re: c * alpha.re + s * beta.im, im: c * alpha.im - s * beta.re },
      beta: { re: c * beta.re + s * alpha.im, im: c * beta.im - s * alpha.re },
    };
  } else if (axis === 'y') {
    return {
      alpha: { re: c * alpha.re - s * beta.re, im: c * alpha.im - s * beta.im },
      beta: { re: c * beta.re + s * alpha.re, im: c * beta.im + s * alpha.im },
    };
  } else {
    const expPlus = Complex.exp(angle / 2);
    const expMinus = Complex.exp(-angle / 2);
    return {
      alpha: Complex.mul(expMinus, alpha),
      beta: Complex.mul(expPlus, beta),
    };
  }
};

// Generate trajectory under Hamiltonian evolution
const generateTrajectory = (initialAlpha, initialBeta, axis, steps = 100, noiseLevel = 0) => {
  const trajectory = [];
  let alpha = { ...initialAlpha };
  let beta = { ...initialBeta };
  const angleStep = (2 * Math.PI) / steps;
  
  for (let i = 0; i <= steps; i++) {
    const bloch = stateToBloch(alpha, beta);
    
    // Add noise as geodesic deviation
    if (noiseLevel > 0) {
      const noise = {
        x: (Math.random() - 0.5) * noiseLevel * 0.1,
        y: (Math.random() - 0.5) * noiseLevel * 0.1,
        z: (Math.random() - 0.5) * noiseLevel * 0.1,
      };
      bloch.x += noise.x;
      bloch.y += noise.y;
      bloch.z += noise.z;
      
      // Renormalize (project back toward sphere, but allow some shrinkage for mixed states)
      const r = Math.sqrt(bloch.x ** 2 + bloch.y ** 2 + bloch.z ** 2);
      const targetR = 1 - noiseLevel * 0.01 * i; // Decoherence shrinks toward center
      if (r > 0.01) {
        const scale = Math.max(0.1, targetR) / r;
        bloch.x *= scale;
        bloch.y *= scale;
        bloch.z *= scale;
      }
    }
    
    trajectory.push({ ...bloch, t: i / steps });
    
    const result = applyRotation(alpha, beta, axis, angleStep);
    alpha = result.alpha;
    beta = result.beta;
  }
  
  return trajectory;
};

// 3D projection with rotation
const project3D = (point, rotation, scale = 120) => {
  // Apply Y rotation
  let x = point.x * Math.cos(rotation.y) + point.z * Math.sin(rotation.y);
  let z = -point.x * Math.sin(rotation.y) + point.z * Math.cos(rotation.y);
  let y = point.y;
  
  // Apply X rotation
  const y2 = y * Math.cos(rotation.x) - z * Math.sin(rotation.x);
  const z2 = y * Math.sin(rotation.x) + z * Math.cos(rotation.x);
  
  return {
    x: x * scale,
    y: y2 * scale,
    z: z2,
  };
};

const BlochSphere = () => {
  const [rotation, setRotation] = useState({ x: 0.4, y: 0.6 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [animationTime, setAnimationTime] = useState(0);
  const [evolutionAxis, setEvolutionAxis] = useState('z');
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [showIdeal, setShowIdeal] = useState(true);
  const [showNoisy, setShowNoisy] = useState(true);
  const [initialState, setInitialState] = useState('plus'); // |+⟩ state
  const animationRef = useRef();
  
  // Initial states
  const initialStates = useMemo(() => ({
    plus: { alpha: { re: 1 / Math.sqrt(2), im: 0 }, beta: { re: 1 / Math.sqrt(2), im: 0 } },
    zero: { alpha: { re: 1, im: 0 }, beta: { re: 0, im: 0 } },
    one: { alpha: { re: 0, im: 0 }, beta: { re: 1, im: 0 } },
    plusI: { alpha: { re: 1 / Math.sqrt(2), im: 0 }, beta: { re: 0, im: 1 / Math.sqrt(2) } },
  }), []);
  
  const currentInitial = initialStates[initialState];
  
  // Generate trajectories
  const idealTrajectory = useMemo(() => 
    generateTrajectory(currentInitial.alpha, currentInitial.beta, evolutionAxis, 100, 0),
    [currentInitial, evolutionAxis]
  );
  
  const noisyTrajectory = useMemo(() => 
    generateTrajectory(currentInitial.alpha, currentInitial.beta, evolutionAxis, 100, noiseLevel),
    [currentInitial, evolutionAxis, noiseLevel]
  );
  
  // Animation loop
  useEffect(() => {
    const animate = () => {
      setAnimationTime(t => (t + 0.005) % 1);
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationRef.current);
  }, []);
  
  // Mouse handlers
  const handleMouseDown = useCallback((e) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, []);
  
  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setRotation(r => ({
      x: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, r.x + dy * 0.01)),
      y: r.y + dx * 0.01,
    }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, [isDragging, lastMouse]);
  
  const handleMouseUp = useCallback(() => setIsDragging(false), []);
  
  // Draw sphere wireframe
  const sphereLines = useMemo(() => {
    const lines = [];
    // Latitude lines
    for (let lat = -60; lat <= 60; lat += 30) {
      const points = [];
      const r = Math.cos(lat * Math.PI / 180);
      const y = Math.sin(lat * Math.PI / 180);
      for (let lon = 0; lon <= 360; lon += 10) {
        points.push({
          x: r * Math.cos(lon * Math.PI / 180),
          y: y,
          z: r * Math.sin(lon * Math.PI / 180),
        });
      }
      lines.push(points);
    }
    // Longitude lines
    for (let lon = 0; lon < 180; lon += 30) {
      const points = [];
      for (let lat = -90; lat <= 90; lat += 10) {
        points.push({
          x: Math.cos(lat * Math.PI / 180) * Math.cos(lon * Math.PI / 180),
          y: Math.sin(lat * Math.PI / 180),
          z: Math.cos(lat * Math.PI / 180) * Math.sin(lon * Math.PI / 180),
        });
      }
      lines.push(points);
    }
    return lines;
  }, []);
  
  // Project and render
  const centerX = 200;
  const centerY = 200;
  
  const projectPoint = (p) => {
    const proj = project3D(p, rotation);
    return { x: centerX + proj.x, y: centerY - proj.y, z: proj.z };
  };
  
  // Axis labels
  const axes = [
    { point: { x: 1.3, y: 0, z: 0 }, label: 'X', color: '#ef4444' },
    { point: { x: -1.3, y: 0, z: 0 }, label: '-X', color: '#ef4444' },
    { point: { x: 0, y: 1.3, z: 0 }, label: '|0⟩', color: '#22c55e' },
    { point: { x: 0, y: -1.3, z: 0 }, label: '|1⟩', color: '#22c55e' },
    { point: { x: 0, y: 0, z: 1.3 }, label: 'Y', color: '#3b82f6' },
    { point: { x: 0, y: 0, z: -1.3 }, label: '-Y', color: '#3b82f6' },
  ];
  
  // Current state position
  const currentIndex = Math.floor(animationTime * (idealTrajectory.length - 1));
  const idealPoint = idealTrajectory[currentIndex];
  const noisyPoint = noisyTrajectory[Math.min(currentIndex, noisyTrajectory.length - 1)];
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white p-6">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-light tracking-wide mb-2" style={{ fontFamily: 'Georgia, serif' }}>
            Bloch Sphere Dynamics
          </h1>
          <p className="text-slate-400 text-lg">
            Geodesic Evolution & Decoherence as Geometric Deviation
          </p>
        </header>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main visualization */}
          <div className="lg:col-span-2 bg-slate-900/50 rounded-2xl p-6 backdrop-blur border border-slate-800">
            <svg
              width="400"
              height="400"
              viewBox="0 0 400 400"
              className="mx-auto cursor-grab active:cursor-grabbing"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <defs>
                <radialGradient id="sphereGradient" cx="30%" cy="30%">
                  <stop offset="0%" stopColor="#1e293b" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#0f172a" stopOpacity="0.95" />
                </radialGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              
              {/* Sphere background */}
              <circle cx={centerX} cy={centerY} r="120" fill="url(#sphereGradient)" stroke="#334155" strokeWidth="1" />
              
              {/* Wireframe */}
              {sphereLines.map((line, i) => (
                <path
                  key={i}
                  d={line.map((p, j) => {
                    const proj = projectPoint(p);
                    return `${j === 0 ? 'M' : 'L'} ${proj.x} ${proj.y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="#475569"
                  strokeWidth="0.5"
                  opacity="0.4"
                />
              ))}
              
              {/* Axes */}
              {[
                { from: { x: -1.2, y: 0, z: 0 }, to: { x: 1.2, y: 0, z: 0 }, color: '#ef4444' },
                { from: { x: 0, y: -1.2, z: 0 }, to: { x: 0, y: 1.2, z: 0 }, color: '#22c55e' },
                { from: { x: 0, y: 0, z: -1.2 }, to: { x: 0, y: 0, z: 1.2 }, color: '#3b82f6' },
              ].map((axis, i) => {
                const from = projectPoint(axis.from);
                const to = projectPoint(axis.to);
                return (
                  <line
                    key={i}
                    x1={from.x} y1={from.y}
                    x2={to.x} y2={to.y}
                    stroke={axis.color}
                    strokeWidth="1.5"
                    opacity="0.6"
                  />
                );
              })}
              
              {/* Axis labels */}
              {axes.map((axis, i) => {
                const proj = projectPoint(axis.point);
                return (
                  <text
                    key={i}
                    x={proj.x}
                    y={proj.y}
                    fill={axis.color}
                    fontSize="12"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontFamily="monospace"
                  >
                    {axis.label}
                  </text>
                );
              })}
              
              {/* Ideal trajectory */}
              {showIdeal && (
                <path
                  d={idealTrajectory.slice(0, currentIndex + 1).map((p, j) => {
                    const proj = projectPoint(p);
                    return `${j === 0 ? 'M' : 'L'} ${proj.x} ${proj.y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="2"
                  filter="url(#glow)"
                  opacity="0.9"
                />
              )}
              
              {/* Noisy trajectory */}
              {showNoisy && noiseLevel > 0 && (
                <path
                  d={noisyTrajectory.slice(0, currentIndex + 1).map((p, j) => {
                    const proj = projectPoint(p);
                    return `${j === 0 ? 'M' : 'L'} ${proj.x} ${proj.y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="#ec4899"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                  opacity="0.8"
                />
              )}
              
              {/* Current ideal state */}
              {showIdeal && idealPoint && (
                <circle
                  cx={projectPoint(idealPoint).x}
                  cy={projectPoint(idealPoint).y}
                  r="8"
                  fill="#f59e0b"
                  filter="url(#glow)"
                />
              )}
              
              {/* Current noisy state */}
              {showNoisy && noiseLevel > 0 && noisyPoint && (
                <circle
                  cx={projectPoint(noisyPoint).x}
                  cy={projectPoint(noisyPoint).y}
                  r="6"
                  fill="#ec4899"
                  stroke="#fff"
                  strokeWidth="1"
                />
              )}
              
              {/* Geodesic deviation vector */}
              {showIdeal && showNoisy && noiseLevel > 0 && idealPoint && noisyPoint && (
                <line
                  x1={projectPoint(idealPoint).x}
                  y1={projectPoint(idealPoint).y}
                  x2={projectPoint(noisyPoint).x}
                  y2={projectPoint(noisyPoint).y}
                  stroke="#a855f7"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                  opacity="0.7"
                />
              )}
            </svg>
            
            <div className="flex justify-center gap-6 mt-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-amber-500" />
                <span className="text-slate-300">Ideal geodesic</span>
              </div>
              {noiseLevel > 0 && (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-pink-500" />
                    <span className="text-slate-300">Noisy trajectory</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-purple-500" style={{ borderStyle: 'dashed' }} />
                    <span className="text-slate-300">Deviation</span>
                  </div>
                </>
              )}
            </div>
          </div>
          
          {/* Controls */}
          <div className="space-y-4">
            <div className="bg-slate-900/50 rounded-2xl p-5 backdrop-blur border border-slate-800">
              <h3 className="text-lg font-medium mb-4 text-slate-200">Evolution Parameters</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Hamiltonian Axis (σ rotation)</label>
                  <div className="flex gap-2">
                    {['x', 'y', 'z'].map(axis => (
                      <button
                        key={axis}
                        onClick={() => setEvolutionAxis(axis)}
                        className={`flex-1 py-2 rounded-lg font-mono text-sm transition-all ${
                          evolutionAxis === axis
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                        }`}
                      >
                        σ{axis.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Initial State</label>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { key: 'plus', label: '|+⟩' },
                      { key: 'zero', label: '|0⟩' },
                      { key: 'one', label: '|1⟩' },
                      { key: 'plusI', label: '|+i⟩' },
                    ].map(state => (
                      <button
                        key={state.key}
                        onClick={() => setInitialState(state.key)}
                        className={`py-2 rounded-lg font-mono text-sm transition-all ${
                          initialState === state.key
                            ? 'bg-emerald-600 text-white'
                            : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                        }`}
                      >
                        {state.label}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm text-slate-400 mb-2">
                    Noise Level (Lindblad strength): {noiseLevel.toFixed(1)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="5"
                    step="0.5"
                    value={noiseLevel}
                    onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                    className="w-full accent-pink-500"
                  />
                </div>
                
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input
                      type="checkbox"
                      checked={showIdeal}
                      onChange={(e) => setShowIdeal(e.target.checked)}
                      className="accent-amber-500"
                    />
                    Show ideal
                  </label>
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input
                      type="checkbox"
                      checked={showNoisy}
                      onChange={(e) => setShowNoisy(e.target.checked)}
                      className="accent-pink-500"
                    />
                    Show noisy
                  </label>
                </div>
              </div>
            </div>
            
            {/* Info panel */}
            <div className="bg-slate-900/50 rounded-2xl p-5 backdrop-blur border border-slate-800">
              <h3 className="text-lg font-medium mb-3 text-slate-200">Geometric Interpretation</h3>
              <div className="text-sm text-slate-400 space-y-3">
                <p>
                  <span className="text-amber-400">Ideal evolution</span> follows geodesics on the Bloch sphere 
                  determined by the Hamiltonian H = ℏω σ<sub>{evolutionAxis.toUpperCase()}</sub>/2.
                </p>
                {noiseLevel > 0 && (
                  <p>
                    <span className="text-pink-400">Noisy evolution</span> shows geodesic deviation from 
                    Lindblad operators. The trajectory spirals inward as purity decreases (mixed states).
                  </p>
                )}
                <p className="text-xs text-slate-500 pt-2 border-t border-slate-800">
                  Drag to rotate • The purple dashed line shows the deviation vector between ideal and noisy states
                </p>
              </div>
            </div>
            
            {/* State readout */}
            <div className="bg-slate-900/50 rounded-2xl p-5 backdrop-blur border border-slate-800 font-mono text-sm">
              <h3 className="text-lg font-medium mb-3 text-slate-200" style={{ fontFamily: 'Georgia, serif' }}>State Coordinates</h3>
              {idealPoint && (
                <div className="space-y-1 text-slate-300">
                  <div>x = <span className="text-amber-400">{idealPoint.x.toFixed(3)}</span></div>
                  <div>y = <span className="text-amber-400">{idealPoint.y.toFixed(3)}</span></div>
                  <div>z = <span className="text-amber-400">{idealPoint.z.toFixed(3)}</span></div>
                  {noiseLevel > 0 && noisyPoint && (
                    <div className="pt-2 mt-2 border-t border-slate-700">
                      <div className="text-pink-400">
                        Purity: {(noisyPoint.x ** 2 + noisyPoint.y ** 2 + noisyPoint.z ** 2).toFixed(3)}
                      </div>
                      <div className="text-purple-400">
                        Deviation: {Math.sqrt(
                          (idealPoint.x - noisyPoint.x) ** 2 +
                          (idealPoint.y - noisyPoint.y) ** 2 +
                          (idealPoint.z - noisyPoint.z) ** 2
                        ).toFixed(3)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Theory section */}
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-slate-900/30 rounded-xl p-5 border border-slate-800/50">
            <h4 className="text-indigo-400 font-medium mb-2">Spectral Connection</h4>
            <p className="text-sm text-slate-400">
              The eigenvalues of ρ determine the state's position: pure states (λ ∈ {`{0,1}`}) lie on the sphere surface, 
              while mixed states (0 &lt; λ &lt; 1) occupy the interior. The entanglement spectrum for bipartite states 
              follows the same logic.
            </p>
          </div>
          <div className="bg-slate-900/30 rounded-xl p-5 border border-slate-800/50">
            <h4 className="text-emerald-400 font-medium mb-2">Fisher-Rao ↔ Fubini-Study</h4>
            <p className="text-sm text-slate-400">
              The natural metric on pure states is the Fubini-Study metric, which equals 4× the Fisher information 
              metric when parameterizing states. Your geometric framework applies directly to quantum parameter 
              estimation.
            </p>
          </div>
          <div className="bg-slate-900/30 rounded-xl p-5 border border-slate-800/50">
            <h4 className="text-pink-400 font-medium mb-2">Decoherence Geometry</h4>
            <p className="text-sm text-slate-400">
              Lindblad operators act as "metric perturbations" causing geodesic deviation. Strong decoherence 
              collapses trajectories toward the center (maximally mixed state ρ = I/2), while weak noise causes 
              gentle spiral decay.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlochSphere;
