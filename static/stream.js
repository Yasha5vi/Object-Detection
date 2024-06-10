function startStream() {
    var video = document.getElementById('videoElement');
    video.src = '/video_feed';
    //videoStreamDiv.appendChild(video);
}

function stop(){
    var video = document.getElementById('videoElement');
    video.src = '';
}