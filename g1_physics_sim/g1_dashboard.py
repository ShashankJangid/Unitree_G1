import http.server
import json
import threading
import time
from typing import Any, Dict, Optional

_telemetry_state: Dict[str, Any] = {
    "timestamp": 0.0,
    "active_mode": "ARM",
    "arm": "RIGHT",
    "target": [0.0, 0.0, 0.0],
    "hand": [0.0, 0.0, 0.0],
    "err_mm": 0.0,
    "euler": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "q": [0.0] * 29,
    "tau": [0.0] * 29,
    "q_des": [0.0] * 29,
    "base_pos": [0.0, 0.0, 0.0],
    "walk_cmd": [0.0, 0.0, 0.0],
    "history": []
}
_state_lock = threading.Lock()
_server_thread: Optional[threading.Thread] = None
_httpd: Optional[http.server.HTTPServer] = None

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unitree G1 Motor & Telemetry Activity Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: #0d1117; color: #c9d1d9; padding: 16px; font-size: 13px; }
  header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-bottom: 16px; }
  h1 { font-size: 18px; color: #58a6ff; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .badge { background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
  .card-title { color: #8b949e; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
  .card-value { font-size: 16px; font-weight: 700; color: #f0f6fc; font-family: monospace; }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; margin-bottom: 16px; }
  .chart-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; position: relative; }
  .chart-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
  .chart-title { font-weight: 600; color: #58a6ff; font-size: 13px; }
  canvas { width: 100% !important; height: 180px !important; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11px; font-family: monospace; }
  th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; }
  .scroll-tbl { max-height: 240px; overflow-y: auto; }
</style>
</head>
<body>
<header>
  <h1><span>Unitree G1</span> Live Motor Activity & Telemetry Monitor</h1>
  <div><span id="connBadge" class="badge">LIVE 50Hz</span></div>
</header>
<div class="grid-stats">
  <div class="card">
    <div class="card-title">Orientation (Yaw / Pitch / Roll)</div>
    <div class="card-value" id="valEuler">Yaw: 0.0° | P: 0.0° | R: 0.0°</div>
  </div>
  <div class="card">
    <div class="card-title">IK Tracking Error</div>
    <div class="card-value" id="valErr" style="color: #3fb950;">0.00 mm</div>
  </div>
  <div class="card">
    <div class="card-title">End-Effector Target vs Hand</div>
    <div class="card-value" id="valPos" style="font-size: 12px;">T: [0, 0, 0] | H: [0, 0, 0]</div>
  </div>
  <div class="card">
    <div class="card-title">Mode & Stance</div>
    <div class="card-value" id="valMode">ARM: RIGHT</div>
  </div>
</div>
<div class="charts-grid">
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Arm Motors Target vs Current (rad)</span>
    </div>
    <canvas id="chartArms"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Leg Joints Positions (rad)</span>
    </div>
    <canvas id="chartLegs"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Real-Time Joint Torques (Nm)</span>
    </div>
    <canvas id="chartTau"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Cartesian Error & Euler Dynamics</span>
    </div>
    <canvas id="chartDyn"></canvas>
  </div>
</div>
<div class="card">
  <div class="card-title">Full 29-Motor Live Status Table</div>
  <div class="scroll-tbl">
    <table id="tblMotors">
      <thead>
        <tr><th>ID</th><th>Joint Name</th><th>q_des (rad)</th><th>q_act (rad)</th><th>Error (rad)</th><th>Tau (Nm)</th></tr>
      </thead>
      <tbody id="tblBody"></tbody>
    </table>
  </div>
</div>
<script>
const JOINT_NAMES = [
  'L_HipPitch', 'L_HipRoll', 'L_HipYaw', 'L_Knee', 'L_AnklePitch', 'L_AnkleRoll',
  'R_HipPitch', 'R_HipRoll', 'R_HipYaw', 'R_Knee', 'R_AnklePitch', 'R_AnkleRoll',
  'WaistYaw', 'WaistRoll', 'WaistPitch',
  'L_ShoulderPitch', 'L_ShoulderRoll', 'L_ShoulderYaw', 'L_Elbow', 'L_WristRoll', 'L_WristPitch', 'L_WristYaw',
  'R_ShoulderPitch', 'R_ShoulderRoll', 'R_ShoulderYaw', 'R_Elbow', 'R_WristRoll', 'R_WristPitch', 'R_WristYaw'
];
const chartCfg = (labelList, colors) => ({
  type: 'line',
  data: {
    labels: Array(50).fill(''),
    datasets: labelList.map((lbl, idx) => ({
      label: lbl,
      data: Array(50).fill(0),
      borderColor: colors[idx % colors.length],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      tension: 0.1
    }))
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { display: false, grid: { display: false } },
      y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 10 } } }
    },
    plugins: { legend: { labels: { color: '#c9d1d9', font: { size: 10 }, boxWidth: 10 } } }
  }
});
const cArms = new Chart(document.getElementById('chartArms'), chartCfg(['R_Sh_Pitch', 'R_Sh_Roll', 'R_Elbow', 'L_Sh_Pitch', 'L_Elbow'], ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff']));
const cLegs = new Chart(document.getElementById('chartLegs'), chartCfg(['L_Knee', 'R_Knee', 'L_HipPitch', 'R_HipPitch', 'L_AnklePitch', 'R_AnklePitch'], ['#3fb950', '#58a6ff', '#f85149', '#d29922', '#bc8cff', '#39d353']));
const cTau = new Chart(document.getElementById('chartTau'), chartCfg(['R_Elbow_Tau', 'L_Elbow_Tau', 'R_Knee_Tau', 'L_Knee_Tau'], ['#f85149', '#d29922', '#58a6ff', '#3fb950']));
const cDyn = new Chart(document.getElementById('chartDyn'), chartCfg(['Err (mm)', 'Yaw (°)', 'Pitch (°)', 'Roll (°)'], ['#f85149', '#58a6ff', '#3fb950', '#d29922']));
const tbody = document.getElementById('tblBody');
for (let i = 0; i < 29; i++) {
  const tr = document.createElement('tr');
  tr.id = 'row_' + i;
  tr.innerHTML = `<td>${i}</td><td>${JOINT_NAMES[i]}</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.00</td>`;
  tbody.appendChild(tr);
}
function pushData(chart, vals) {
  chart.data.datasets.forEach((ds, i) => {
    ds.data.push(vals[i] !== undefined ? vals[i] : 0);
    ds.data.shift();
  });
  chart.update();
}
async function poll() {
  try {
    const res = await fetch('/api/telemetry');
    if (!res.ok) return;
    const d = await res.json();
    document.getElementById('valEuler').innerText = `Yaw: ${d.euler.yaw.toFixed(1)}° | P: ${d.euler.pitch.toFixed(1)}° | R: ${d.euler.roll.toFixed(1)}°`;
    const errEl = document.getElementById('valErr');
    errEl.innerText = `${d.err_mm.toFixed(2)} mm`;
    errEl.style.color = d.err_mm < 5.0 ? '#3fb950' : '#f85149';
    document.getElementById('valPos').innerText = `T:[${d.target.map(v => v.toFixed(2)).join(',')}] | H:[${d.hand.map(v => v.toFixed(2)).join(',')}]`;
    document.getElementById('valMode').innerText = `${d.active_mode} (${d.arm})`;
    pushData(cArms, [d.q[22], d.q[23], d.q[25], d.q[15], d.q[18]]);
    pushData(cLegs, [d.q[3], d.q[9], d.q[0], d.q[6], d.q[4], d.q[10]]);
    pushData(cTau, [d.tau[25], d.tau[18], d.tau[9], d.tau[3]]);
    pushData(cDyn, [d.err_mm, d.euler.yaw, d.euler.pitch, d.euler.roll]);
    for (let i = 0; i < 29; i++) {
      const r = document.getElementById('row_' + i);
      if (r) {
        const qDes = (d.q_des && d.q_des[i] !== undefined) ? d.q_des[i] : 0;
        const qAct = (d.q && d.q[i] !== undefined) ? d.q[i] : 0;
        const err = qDes - qAct;
        const tau = (d.tau && d.tau[i] !== undefined) ? d.tau[i] : 0;
        r.cells[2].innerText = qDes.toFixed(3);
        r.cells[3].innerText = qAct.toFixed(3);
        r.cells[4].innerText = err.toFixed(3);
        r.cells[5].innerText = tau.toFixed(2);
      }
    }
  } catch(e) {}
}
setInterval(poll, 60);
</script>
</body>
</html>
"""

class G1TelemetryHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        elif self.path == '/api/telemetry':
            with _state_lock:
                payload = json.dumps(_telemetry_state).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def update_telemetry(
    active_mode: str,
    arm: str,
    target: Any,
    hand: Any,
    err_mm: float,
    euler: Dict[str, float],
    q: Any,
    tau: Any,
    q_des: Any,
    base_pos: Any = None,
    walk_cmd: Any = None
) -> None:
    global _telemetry_state
    with _state_lock:
        _telemetry_state["timestamp"] = time.time()
        _telemetry_state["active_mode"] = str(active_mode)
        _telemetry_state["arm"] = str(arm)
        _telemetry_state["target"] = [float(x) for x in target] if target is not None else [0.0, 0.0, 0.0]
        _telemetry_state["hand"] = [float(x) for x in hand] if hand is not None else [0.0, 0.0, 0.0]
        _telemetry_state["err_mm"] = float(err_mm)
        _telemetry_state["euler"] = {k: float(v) for k, v in euler.items()}
        _telemetry_state["q"] = [float(x) for x in q[:29]] if q is not None else [0.0] * 29
        _telemetry_state["tau"] = [float(x) for x in tau[:29]] if tau is not None else [0.0] * 29
        _telemetry_state["q_des"] = [float(x) for x in q_des[:29]] if q_des is not None else [0.0] * 29
        if base_pos is not None:
            _telemetry_state["base_pos"] = [float(x) for x in base_pos[:3]]
        if walk_cmd is not None:
            _telemetry_state["walk_cmd"] = [float(x) for x in walk_cmd[:3]]

def start_dashboard_server(port: int = 8080) -> Optional[http.server.HTTPServer]:
    global _server_thread, _httpd
    if _httpd is not None:
        return _httpd
    try:
        _httpd = http.server.ThreadingHTTPServer(('0.0.0.0', port), G1TelemetryHandler)
        _server_thread = threading.Thread(target=_httpd.serve_forever, daemon=True)
        _server_thread.start()
        print(f"\\n[DASHBOARD] Live motor telemetry dashboard running at http://localhost:{port}")
        return _httpd
    except Exception as ex:
        print(f"\\n[DASHBOARD] Could not bind port {port}: {ex}")
        return None
