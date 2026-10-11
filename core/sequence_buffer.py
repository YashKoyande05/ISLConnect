import collections

class SequenceBuffer:
    """
    Manages a temporal buffer of recent predictions to stabilize continuous signing 
    and extract meaningful words or glosses from a stream of frames.
    """
    def __init__(self, buffer_size=15, threshold=0.7):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.predictions = collections.deque(maxlen=buffer_size)
        self.confidences = collections.deque(maxlen=buffer_size)
        
        self.current_gloss = None
        self.gloss_sequence = []
        
    def add_prediction(self, prediction, confidence):
        if not prediction:
            self.predictions.append(None)
            self.confidences.append(0.0)
            return None
            
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        
        return self._evaluate_buffer()
        
    def _evaluate_buffer(self):
        """
        Evaluate the sliding window. If a sign dominates the buffer with high confidence,
        we emit it as a recognized gloss, ensuring we don't emit the same sign twice in a row.
        """
        if len(self.predictions) < self.buffer_size:
            return None
            
        # Count non-None predictions
        valid_preds = [p for p in self.predictions if p is not None]
        if not valid_preds:
            return None
            
        # Find the most common prediction in the buffer
        counter = collections.Counter(valid_preds)
        most_common, count = counter.most_common(1)[0]
        
        # Calculate ratio of most common prediction in the entire buffer
        ratio = count / self.buffer_size
        
        # Calculate average confidence for the most common prediction
        avg_conf = sum(c for p, c in zip(self.predictions, self.confidences) if p == most_common) / count
        
        # If it passes our temporal threshold and confidence threshold
        if ratio >= 0.6 and avg_conf >= self.threshold:
            if most_common != self.current_gloss:
                self.current_gloss = most_common
                self.gloss_sequence.append(most_common)
                return most_common
                
        # If the buffer is empty (user lowered hands), reset current gloss
        if ratio < 0.2:
            self.current_gloss = None
            
        return None
        
    def get_sequence(self):
        return self.gloss_sequence
        
    def clear(self):
        self.predictions.clear()
        self.confidences.clear()
        self.current_gloss = None
        self.gloss_sequence = []
