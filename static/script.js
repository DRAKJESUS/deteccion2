document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData();
    const imageInput = document.getElementById("imageInput").files[0];
    formData.append("file", imageInput);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
    } else {
        document.getElementById("outputImage").src = `data:image/png;base64,${data.image_with_points_base64}`;
        document.getElementById("outputImage").style.display = "block";
        document.getElementById("emotion").textContent = `Emoción principal: ${data.dominant_emotion}`;
    }
});
