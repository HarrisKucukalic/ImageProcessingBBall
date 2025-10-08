def update_track(all_tracks, unmatches_track, offline_all_tracks, kf, max_age=30):
    """
    Updates all active tracks.
    - Predicts the next state for each track using the Kalman filter.
    - Increments the age of each track.
    - Moves tracks that are older than max_age to the offline list.
    - Adds newly created tracks to the active list.
    """
    # Predict the next state for all existing tracks and increment their age
    for track in all_tracks:
        # The inc_count() should happen for every frame the track exists but is not matched.
        # This logic is handled in the matching_assignment step where matched tracks have their counter reset.
        # Here, we just increment the age for everyone since this is a new frame.
        track.inc_count()

        # Predict the next location
        pred_m, pred_c = kf.predict(track.mean[-1], track.covariance)
        track.mean.append(pred_m)
        track.covariance = pred_c

    # --- Prune Old Tracks ---
    # Move tracks that have been unmatched for too long to the offline list
    for track in all_tracks:
        if track.counter > max_age:
            offline_all_tracks.append(track)

    # Create a new list of active tracks, keeping only the ones that are not too old.
    # This is the safe way to "delete" items from a list.
    active_tracks = [t for t in all_tracks if t.counter <= max_age]

    # Add the newly created tracks from the current frame to the active list
    active_tracks.extend(unmatches_track)

    return active_tracks, offline_all_tracks