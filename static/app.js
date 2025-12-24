const chatForm = document.getElementById("chatForm");
const chatBox = document.getElementById("chatBox");

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const input = document.getElementById("userInput");
    const question = input.value;

    chatBox.innerHTML += `<div class="user-message">${question}</div>`;
    input.value = "";

    const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
    });

    const data = await res.json();

    chatBox.innerHTML += `<div class="ai-message">${data.answer}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
});
