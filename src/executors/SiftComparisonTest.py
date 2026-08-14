def _build_visualization(
    self,
    keypoints1_dicts,
    keypoints2_dicts,
    good_matches
):

    # İki SIFT çıktısındaki maksimum koordinatları bul
    max_x1 = max(
        [kp["pt"][0] for kp in keypoints1_dicts],
        default=0
    )

    max_y1 = max(
        [kp["pt"][1] for kp in keypoints1_dicts],
        default=0
    )

    max_x2 = max(
        [kp["pt"][0] for kp in keypoints2_dicts],
        default=0
    )

    max_y2 = max(
        [kp["pt"][1] for kp in keypoints2_dicts],
        default=0
    )

    # Canvas boyutları
    height = int(
        max(max_y1, max_y2) + 100
    )

    width1 = int(
        max_x1 + 100
    )

    width2 = int(
        max_x2 + 100
    )

    gap = 100

    total_width = (
        width1
        + width2
        + gap
    )

    canvas = np.ones(
        (height, total_width, 3),
        dtype=np.uint8
    ) * 255

    # --------------------------------------------------------
    # KEYPOINTLER
    # --------------------------------------------------------

    for kp in keypoints1_dicts:

        x = int(kp["pt"][0])
        y = int(kp["pt"][1])

        cv2.circle(
            canvas,
            (x, y),
            4,
            (255, 0, 0),
            -1
        )

    for kp in keypoints2_dicts:

        x = int(
            kp["pt"][0]
            + width1
            + gap
        )

        y = int(
            kp["pt"][1]
        )

        cv2.circle(
            canvas,
            (x, y),
            4,
            (255, 0, 0),
            -1
        )

    # --------------------------------------------------------
    # MATCH CONNECTIONS
    # --------------------------------------------------------

    for match in good_matches:

        kp1 = keypoints1_dicts[
            match.queryIdx
        ]

        kp2 = keypoints2_dicts[
            match.trainIdx
        ]

        p1 = (
            int(kp1["pt"][0]),
            int(kp1["pt"][1])
        )

        p2 = (
            int(kp2["pt"][0] + width1 + gap),
            int(kp2["pt"][1])
        )

        cv2.line(
            canvas,
            p1,
            p2,
            (0, 255, 0),
            1
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "SIFT 1",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        "SIFT 2",
        (
            width1 + gap + 20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    return canvas