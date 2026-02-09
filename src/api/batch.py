import time

def process_batch_items(predictor, analytics_engine, items):
    """
    Processes a list of items (URLs or Text) and returns aggregated results.
    """
    results = []
    start_time = time.time()
    
    for item in items:
        item_type = "URL" if item.startswith('http') else "Text"
        
        # This is a simplified simulation of the predictor/analytics flow
        # In a real app, we'd call the predictor.predict() and analytics functions
        # For now, we reuse the logic from predict_api to return a list
        results.append({
            "item": item[:50] + "..." if len(item) > 50 else item,
            "type": item_type,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
    execution_time = round(time.time() - start_time, 2)
    
    return {
        "batch_id": int(time.time()),
        "total_items": len(items),
        "results": results,
        "execution_time": execution_time
    }
