# Cải thiện model với data drift không có label bằng Unsupervised Learning

## Ý tưởng

-   Ý tưởng của phương pháp này dựa trên data đã có label để gán label cho data drift bằng cách sử dụng Unsupervised Learning.
-   Đầu tiên, chúng ta sử dụng data ban đầu đã có label để xây dựng một clustering model chia data thành N clusters khác nhau.
-   Clustering model sau đó được sử dụng để tìm ra cluster tương đồng nhất cho từng data point trong bộ data drift.
-   Với các bài toán Regression, label của từng data point trong bộ data drift sẽ được tính bằng giá trị trung bình label của các data points trong bộ data gốc thuộc cluster đã tìm được.
-   Với các bài toán Classification, label của từng data point trong bộ data drift sẽ là label chiếm đa số trong các label của các data points trong bộ data gốc thuộc cluster đã tìm được.
-   Sử dụng các label tìm được cho data drift để train model mới.

## Các bước thực hiện

1. Xây dựng model clustering cho data đã có label

    ```python
    from sklearn.cluster import MiniBatchKMeans

    # N là số lượng cluster
    N = 10000 * len(np.unique(normal_train_y))
    # Train clustering model cho data đã có label
    kmeans = MiniBatchKMeans(n_clusters=N, random_state=0).fit(normal_train_x)
    ```

2. Predict cluster cho từng data point trong bộ data drift

    ```python
    drift_train_clusters = kmeans.predict(drift_train_x)
    ```

3. Tìm label mới cho bộ data drift

    ```python
    # Tạo 1 mảng ánh xạ cluster với 1 label mới (do các data drift thuộc cùng 1 cluster sẽ có label giống nhau)
    new_labels = []

    # Duyệt từng cluster
    for  i  in  range(N):
    	# Lấy các label của các data point thuộc cluster i
    	mask = (kmeans.labels_ == i)
    	cluster_labels = normal_train_y[mask]

    	if  len(cluster_labels) == 0:
    		# Nếu cluster i rỗng thì xác định cluster i ánh xạ với 1 label mặc định (ở đây lựa chọn là 0)
    		new_labels.append(0)
    	else:
    		# Tìm label mới cho cả cụm cluster trong trường hợp cụm cluster khác rỗng
    		if  isinstance(normal_train_y.flatten()[0], float):
    			# Nếu là bài toán Regression thì lấy giá trị trung bình của các label thuộc cluster
    			new_labels.append(np.mean(cluster_labels.flatten()))
    		else:
    			# Nếu là bài toán Classification thì lấy label xuất hiện nhiều nhất trong cluster
    			new_labels.append(np.bincount(cluster_labels.flatten()).argmax())

    # Ánh xạ lại label cho data drift dựa trên kết quả predict cluster ở trên
    y_drift_propagated = [new_labels[c] for  c  in  drift_train_clusters]
    ```

4. Training model mới với label mới của bộ data drift

    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    drift_model = make_pipeline(StandardScaler(), RandomForestClassifier())

    drift_model.fit(drift_train_x, y_drift_propagated)
    ```
