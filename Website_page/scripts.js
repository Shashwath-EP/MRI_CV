// scripts.js

document.getElementById("predictButton").addEventListener("click", function () {
    const imageUpload = document.getElementById("imageUpload").files[0];
    const modelSelect = document.getElementById("modelSelect").value;

    if (!imageUpload) {
        alert("Please upload an image.");
        return;
    }

    // Create FormData object to hold the uploaded file and model choice
    const formData = new FormData();
    formData.append("image", imageUpload);
    formData.append("model_name", modelSelect);

    // Show the uploaded image
    const reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("uploadedImage").src = e.target.result;
    };
    reader.readAsDataURL(imageUpload);

    // Send request to the FastAPI backend
    fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Convert the mask bytes to an image and display it
        const maskBytes = data.prediction;
        const maskBlob = new Blob([new Uint8Array(maskBytes)], { type: "image/png" });
        const maskUrl = URL.createObjectURL(maskBlob);
        document.getElementById("predictedMask").src = maskUrl;

        // Show results section
        document.getElementById("resultSection").style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred during prediction.");
    });
});
