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
