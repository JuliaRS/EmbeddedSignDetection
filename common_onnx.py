import onnxruntime.backend as backend
import onnx

def get_executor(model, device='CPU'):
    beckend_rep = backend.prepare(model=str(model))
    sess = beckend_rep._session
    inputs_info = sess.get_inputs()
    input_names = [input_layer.name for input_layer in inputs_info]
    outputs = sess.get_outputs()
    output_names = [output.name for output in outputs]
    return sess, input_names, output_names


class ONNXModel:
    def __init__(self, model, device):
        self.model = model
        self.session, self.input_names, self.output_names = get_executor(model, device)

    def infer(self, input_data):
        feed_dict = dict(zip(self.input_names, input_data))
        return self.session.run(self.output_names, feed_dict)
