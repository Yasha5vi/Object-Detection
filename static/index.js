function uploadImage() {
    var input = document.getElementById('uploadForm');
    var file = input.files[0];

    if (file) {
        var formData = new FormData();
        formData.append('file', file);

        fetch('/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.text();
            }
            throw new Error('Network response was not ok.');
        })
        .then(data => {
            console.log(data); // Log success message or handle it as needed
            // Optionally, update the UI with the result, if needed
            // For example, display the processed image:
            var img = document.getElementById('resultImage');
            img.src = 'static/result.jpg';
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });
    }
}
