const WS_URL = "ws://127.0.0.1:8000/ws/traduzir";
const INTERVALO_MS = 100; // Ritmo máximo de verificação para alimentar a janela de 30 frames
const LARGURA_ENVIO = 640;
const QUALIDADE_JPEG = 0.7;

const btnIniciar = document.getElementById("btn-iniciar");
const btnParar = document.getElementById("btn-parar");
const videoEl = document.getElementById("video");
const videoVazioEl = document.getElementById("video-vazio");
const saidaEl = document.getElementById("saida");

// Canvas fora da tela para extrair frames do <video> sem mexer no layout
const canvas = document.createElement("canvas");
canvas.width = LARGURA_ENVIO;
canvas.height = 480;
const ctx = canvas.getContext("2d");

let stream = null;
let socket = null;
let loopEnvio = null;
let traducaoAtiva = false;
let encerramentoIntencional = false;
let aguardandoResposta = false;
let tempoUltimoEnvio = 0;


function definirSaida(html, estado = "normal") {
  if (!saidaEl) return;
  saidaEl.dataset.estado = estado;
  saidaEl.innerHTML = html;
}

function escaparHtml(texto) {
  const div = document.createElement("div");
  div.textContent = String(texto ?? "");
  return div.innerHTML;
}

function falarTexto(texto) {
  if (!("speechSynthesis" in window)) {
    console.warn("Este navegador não suporta síntese de voz.");
    return;
  }

  const fala = new SpeechSynthesisUtterance(texto);
  fala.lang = "pt-BR";
  fala.rate = 1;
  fala.pitch = 1;
  fala.volume = 1;

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(fala);
}

function atualizarBotao(ativo) {
  if (btnIniciar) {
    btnIniciar.disabled = ativo;
    btnIniciar.textContent = ativo ? "Traduzindo..." : "Iniciar Tradução";
    btnIniciar.classList.toggle("ativo", ativo);
  }

  if (btnParar) {
    btnParar.disabled = !ativo;
  }
}

async function iniciarCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Seu navegador não suporta acesso à câmera (getUserMedia).");
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  videoEl.srcObject = stream;
  await videoEl.play();

  // Ajusta a altura do canvas mantendo a proporção real da câmera
  if (videoEl.videoWidth && videoEl.videoHeight) {
    const proporcao = videoEl.videoHeight / videoEl.videoWidth;
    canvas.width = LARGURA_ENVIO;
    canvas.height = Math.round(LARGURA_ENVIO * proporcao);
  }

  videoEl.classList.add("ligado");
  if (videoVazioEl) {
    videoVazioEl.style.display = "none";
  }
}

function pararCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  if (videoEl) {
    videoEl.pause();
    videoEl.srcObject = null;
    videoEl.classList.remove("ligado");
  }

  if (videoVazioEl) {
    videoVazioEl.style.display = "";
  }
}

function enviarProximoFrame() {
  if (!traducaoAtiva || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Libera a trava de segurança se demorar mais de 2s sem resposta
  if (aguardandoResposta && performance.now() - tempoUltimoEnvio > 2000) {
    aguardandoResposta = false;
  }

  if (aguardandoResposta || socket.bufferedAmount > 256 * 1024) {
    return;
  }

  if (!videoEl || videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  aguardandoResposta = true;
  tempoUltimoEnvio = performance.now();

  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    (blob) => {
      if (blob && socket && socket.readyState === WebSocket.OPEN && traducaoAtiva) {
        socket.send(blob);
      } else {
        aguardandoResposta = false;
      }
    },
    "image/jpeg",
    QUALIDADE_JPEG
  );
}

function iniciarLoopFrames() {
  pararLoopFrames();
  aguardandoResposta = false;
  enviarProximoFrame();

  loopEnvio = setInterval(() => {
    enviarProximoFrame();
  }, INTERVALO_MS);
}

function pararLoopFrames() {
  if (loopEnvio) {
    clearInterval(loopEnvio);
    loopEnvio = null;
  }
}

function renderizarTraducao(payload) {
  const sinal = payload?.sinal || "Aguardando sinal...";
  const confianca =
    typeof payload?.confianca === "number" && payload.confianca > 0
      ? ` (${payload.confianca.toFixed(1)}%)`
      : "";
  const textoVoz = payload?.texto_voz ? `<small>${escaparHtml(payload.texto_voz)}</small>` : "";
  if (
    payload?.novo_sinal === true &&
    typeof payload?.texto_voz === "string" &&
    payload.texto_voz.trim()
  ) {
    falarTexto(payload.texto_voz.trim());
  }

  definirSaida(
    `
      <p><strong>${escaparHtml(sinal)}${escaparHtml(confianca)}</strong></p>
      ${textoVoz}
    `,
    "traduzindo"
  );
}

function conectarWebSocket() {
  return new Promise((resolve, reject) => {
    encerramentoIntencional = false;
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "blob";

    let abriu = false;

    ws.onopen = () => {
      abriu = true;
      resolve(ws);
    };

    ws.onmessage = (event) => {
      aguardandoResposta = false;
      try {
        const dados = JSON.parse(event.data);
        renderizarTraducao(dados);
      } catch (err) {
        console.error("Erro ao interpretar resposta do servidor:", err);
      }
      enviarProximoFrame();
    };

    ws.onerror = () => {
      if (!abriu) {
        reject(new Error("Não foi possível conectar ao servidor em ws://127.0.0.1:8000/ws/traduzir."));
      }
    };

    ws.onclose = (event) => {
      aguardandoResposta = false;
      pararLoopFrames();

      if (!encerramentoIntencional && traducaoAtiva) {
        traducaoAtiva = false;
        atualizarBotao(false);
        pararCamera();
        definirSaida(
          `<p>Conexão com o servidor encerrada (código ${event.code}). Verifique o FastAPI e tente novamente.</p>`,
          "erro"
        );
      }
    };
  });
}

async function iniciarTraducao() {
  if (traducaoAtiva) return;

  if (btnIniciar) btnIniciar.disabled = true;
  if (btnParar) btnParar.disabled = true;
  definirSaida("<p>Ativando câmera e conectando ao servidor...</p>", "conectando");

  try {
    await iniciarCamera();
    socket = await conectarWebSocket();
    traducaoAtiva = true;
    atualizarBotao(true);
    definirSaida("<p>Câmera ativa. Faça um sinal em Libras...</p>", "aguardando");
    iniciarLoopFrames();
  } catch (erro) {
    console.error("Falha ao iniciar tradução:", erro);
    pararTraducao("erro", erro.message || "Erro ao acessar a câmera ou o servidor.");
  } finally {
    if (!traducaoAtiva && btnIniciar) {
      btnIniciar.disabled = false;
    }
  }
}

function pararTraducao(tipo = "normal", mensagemCustomizada = "") {
  encerramentoIntencional = true;
  traducaoAtiva = false;
  aguardandoResposta = false;
  pararLoopFrames();

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  if (socket) {
    socket.onclose = null;
    socket.onerror = null;
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close(1000, "Encerrado pelo usuário");
    }
    socket = null;
  }

  pararCamera();
  atualizarBotao(false);

  if (tipo === "erro") {
    definirSaida(`<p>${escaparHtml(mensagemCustomizada)}</p>`, "erro");
  } else {
    definirSaida("<p>Tradução pausada. Faça um sinal em Libras...</p>", "normal");
  }
}

if (btnIniciar) {
  btnIniciar.addEventListener("click", () => {
    if (!traducaoAtiva) {
      iniciarTraducao();
    }
  });
}

if (btnParar) {
  btnParar.addEventListener("click", () => {
    if (traducaoAtiva) {
      pararTraducao("normal");
    }
  });
}

// Garante liberação da câmera e do socket ao fechar/recarregar a aba
window.addEventListener("beforeunload", () => {
  pararTraducao("normal");
});