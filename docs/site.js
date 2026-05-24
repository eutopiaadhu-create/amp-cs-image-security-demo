const roleContent = {
  guest: {
    label: "访客模式",
    title: "只能查看视觉正常的隐写载体",
    body: "适合公开传输或展示场景。普通观察者看不到明显密文，也无法恢复隐藏图像。",
    scope: "隐写后载体图",
    value: "隐藏信息存在性，降低暴露感"
  },
  "user-a": {
    label: "A 用户模式",
    title: "恢复目标图，并可叠加 AI 隐私保护",
    body: "A 用户拥有部分授权，可以恢复目标图；低权限视图可结合 YOLO 对敏感区域进行马赛克保护。",
    scope: "目标图 + 隐私保护",
    value: "适合部分授权查看场景"
  },
  "user-b": {
    label: "B 用户模式",
    title: "恢复源图和目标图，展示完整授权能力",
    body: "B 用户拥有完整授权，可从隐写载体中恢复源图 measurement 和目标图 measurement，并通过 AMP-Net 重建。",
    scope: "源图 + 目标图",
    value: "适合完整授权接收场景"
  }
};

const panel = document.querySelector(".access-panel");
const buttons = document.querySelectorAll("[data-role-button]");
const label = document.querySelector("#role-label");
const title = document.querySelector("#role-title");
const body = document.querySelector("#role-body");
const scope = document.querySelector("#role-scope");
const value = document.querySelector("#role-value");

function setRole(role) {
  const content = roleContent[role];
  if (!content || !panel) return;

  panel.dataset.role = role;
  label.textContent = content.label;
  title.textContent = content.title;
  body.textContent = content.body;
  scope.textContent = content.scope;
  value.textContent = content.value;

  buttons.forEach((button) => {
    const selected = button.dataset.roleButton === role;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  });
}

buttons.forEach((button) => {
  button.addEventListener("click", () => setRole(button.dataset.roleButton));
});

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const packageState = {
  blobUrl: null,
  parsedPackage: null
};

function setStatus(element, message, type = "") {
  if (!element) return;
  element.textContent = message;
  element.classList.toggle("is-ok", type === "ok");
  element.classList.toggle("is-error", type === "error");
}

function bytesToBase64(bytes) {
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
}

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function randomBase64(length) {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return bytesToBase64(bytes);
}

async function deriveKey(password, saltBase64) {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    encoder.encode(password),
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  return crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: base64ToBytes(saltBase64),
      iterations: 120000,
      hash: "SHA-256"
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function fileToDataUrl(file, maxSide = 720) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.86);
}

async function encryptPayload(payload, password) {
  const salt = randomBase64(16);
  const iv = randomBase64(12);
  const key = await deriveKey(password, salt);
  const cipher = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: base64ToBytes(iv) },
    key,
    encoder.encode(JSON.stringify(payload))
  );
  return {
    algorithm: "PBKDF2-SHA256 + AES-256-GCM",
    salt,
    iv,
    ciphertext: bytesToBase64(new Uint8Array(cipher))
  };
}

async function decryptPayload(encrypted, password) {
  const key = await deriveKey(password, encrypted.salt);
  const plain = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: base64ToBytes(encrypted.iv) },
    key,
    base64ToBytes(encrypted.ciphertext)
  );
  return JSON.parse(decoder.decode(plain));
}

function selectedFiles(input, maxCount) {
  return Array.from(input.files || []).slice(0, maxCount);
}

function renderStructurePreview(pkg) {
  return JSON.stringify({
    format: pkg.format,
    version: pkg.version,
    createdAt: pkg.createdAt,
    public: pkg.public,
    encrypted: {
      targetForUserA: {
        algorithm: pkg.encrypted.targetForUserA.algorithm,
        ciphertextBytes: Math.round(pkg.encrypted.targetForUserA.ciphertext.length * 0.75)
      },
      bundleForUserB: {
        algorithm: pkg.encrypted.bundleForUserB.algorithm,
        ciphertextBytes: Math.round(pkg.encrypted.bundleForUserB.ciphertext.length * 0.75)
      }
    }
  }, null, 2);
}

function renderImages(container, images, prefix) {
  container.innerHTML = "";
  images.forEach((image, index) => {
    const card = document.createElement("figure");
    card.className = "output-card";
    const img = document.createElement("img");
    img.src = image.dataUrl;
    img.alt = `${prefix} ${index + 1}`;
    const caption = document.createElement("span");
    caption.textContent = `${prefix} ${index + 1} · ${image.name || "image"}`;
    card.append(img, caption);
    container.append(card);
  });
}

const sourceInput = document.querySelector("#sourceFiles");
const targetInput = document.querySelector("#targetFiles");
const carrierInput = document.querySelector("#carrierFile");
const keyAInput = document.querySelector("#keyA");
const keyBInput = document.querySelector("#keyB");
const generateButton = document.querySelector("#generatePackage");
const downloadLink = document.querySelector("#downloadPackage");
const encryptStatus = document.querySelector("#encryptStatus");
const packagePreview = document.querySelector("#packagePreview");
const packageFileInput = document.querySelector("#packageFile");
const decryptRole = document.querySelector("#decryptRole");
const decryptKey = document.querySelector("#decryptKey");
const decryptButton = document.querySelector("#decryptPackage");
const decryptStatus = document.querySelector("#decryptStatus");
const decryptOutput = document.querySelector("#decryptOutput");

generateButton?.addEventListener("click", async () => {
  try {
    const sourceFiles = selectedFiles(sourceInput, 3);
    const targetFiles = selectedFiles(targetInput, 3);
    const carrierFile = carrierInput.files?.[0];
    const keyA = keyAInput.value.trim();
    const keyB = keyBInput.value.trim();

    if (!sourceFiles.length || !targetFiles.length || !carrierFile) {
      throw new Error("请至少选择 1 张源图、1 张目标图和 1 张载体图。");
    }
    if (!keyA || !keyB) {
      throw new Error("请填写 A 用户密钥和 B 用户密钥。");
    }

    setStatus(encryptStatus, "正在读取图片并生成加密结构...");

    const sources = await Promise.all(sourceFiles.map(async (file) => ({
      name: file.name,
      dataUrl: await fileToDataUrl(file)
    })));
    const targets = await Promise.all(targetFiles.map(async (file) => ({
      name: file.name,
      dataUrl: await fileToDataUrl(file)
    })));
    const carrier = {
      name: carrierFile.name,
      dataUrl: await fileToDataUrl(carrierFile, 960)
    };

    const pkg = {
      format: "AMP-CS-Web-Package",
      version: 1,
      createdAt: new Date().toISOString(),
      public: {
        carrier,
        counts: {
          sourceImages: sources.length,
          targetImages: targets.length
        },
        note: "Public carrier preview is visible to guest users. Encrypted payload requires role keys."
      },
      encrypted: {
        targetForUserA: await encryptPayload({ targets }, keyA),
        bundleForUserB: await encryptPayload({ sources, targets }, keyB)
      }
    };

    const blob = new Blob([JSON.stringify(pkg, null, 2)], { type: "application/json" });
    if (packageState.blobUrl) URL.revokeObjectURL(packageState.blobUrl);
    packageState.blobUrl = URL.createObjectURL(blob);
    downloadLink.href = packageState.blobUrl;
    downloadLink.download = `ampcs-package-${Date.now()}.ampcs.json`;
    downloadLink.classList.remove("is-disabled");
    packagePreview.textContent = renderStructurePreview(pkg);
    setStatus(encryptStatus, "已生成加密包。访客只能查看载体，A/B 用户需使用对应密钥解密。", "ok");
  } catch (error) {
    setStatus(encryptStatus, error.message || "生成失败。", "error");
  }
});

packageFileInput?.addEventListener("change", async () => {
  const file = packageFileInput.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (parsed.format !== "AMP-CS-Web-Package") {
      throw new Error("这不是 AMP-CS Web 加密包。");
    }
    packageState.parsedPackage = parsed;
    renderImages(decryptOutput, [parsed.public.carrier], "访客载体");
    setStatus(decryptStatus, "加密包已导入。当前显示访客可见载体。", "ok");
  } catch (error) {
    packageState.parsedPackage = null;
    decryptOutput.innerHTML = "";
    setStatus(decryptStatus, error.message || "加密包读取失败。", "error");
  }
});

decryptButton?.addEventListener("click", async () => {
  try {
    const pkg = packageState.parsedPackage;
    const role = decryptRole.value;
    const password = decryptKey.value.trim();
    if (!pkg) throw new Error("请先导入 .ampcs.json 加密包。");
    if (!password) throw new Error("请输入对应角色密钥。");

    setStatus(decryptStatus, "正在解密...");
    if (role === "userA") {
      const payload = await decryptPayload(pkg.encrypted.targetForUserA, password);
      renderImages(decryptOutput, payload.targets, "A 用户目标图");
      setStatus(decryptStatus, "A 用户解密成功：已恢复目标图。", "ok");
    } else {
      const payload = await decryptPayload(pkg.encrypted.bundleForUserB, password);
      renderImages(decryptOutput, [...payload.sources, ...payload.targets], "B 用户恢复图");
      setStatus(decryptStatus, "B 用户解密成功：已恢复源图和目标图。", "ok");
    }
  } catch (error) {
    setStatus(decryptStatus, "解密失败：请检查角色和密钥是否匹配。", "error");
  }
});
