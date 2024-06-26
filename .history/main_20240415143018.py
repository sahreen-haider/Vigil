from inference import InferencePipeline

from inference.core.inferences.stream.sinks import render_boxes


pipeline = InferencePipeline.init(
    model_id = "rock-paper-scissors-sxsw/11",
    video_reference=0,
    on_prediction = render_boxes
)

pipeline.start()
pipeline.join()