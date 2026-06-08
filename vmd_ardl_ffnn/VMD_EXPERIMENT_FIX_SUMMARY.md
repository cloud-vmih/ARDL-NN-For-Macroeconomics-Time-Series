# Tong Hop Van De Va Thay Doi VMD Experiment

## 1. Van de quan sat duoc

Khi tang `--vmd-modes`, sai so cua pipeline VMD co xu huong te hon. VMD modes cao hon co the lam model kho hon do so component nhieu hon, nhieu nhieu hon, va moi component co tin hieu yeu hon. Tuy nhien, qua kiem tra code, co mot loi correctness trong qua trinh experiment lam hien tuong nay bi khuech dai va co the dan den chon model sai.

File VMD decomposition khong bi fixed cung `K=3`. Trong `decomposition.py`, so modes da duoc lay tu `self.config.modes`, nen `--vmd-modes` thuc su co tac dung.

## 2. Nguyen nhan chinh trong experiment

Trong pipeline VMD cu, moi candidate FFNN cho tung component duoc du bao ra `predicted_component`, nhung khi cham diem validation lai bi so voi actual target tong.

Cu the:

- `VMD1`, `VMD2`, ..., `RES` la cac component rieng.
- Moi model component chi du bao mot thanh phan cua target.
- Nhung `_score_component_prediction()` cu reconstruct tu mot component duy nhat roi so voi `actual_raw_transformed`.
- Dieu nay lam `Val RMSE`, `Val MAE`, `Val MAPE` cua tung component khong phan anh kha nang du bao component do.

He qua:

- Ranking component candidate bi lech.
- Target-lag validation screen cho component cung bi lech.
- Khi `vmd-modes` tang, so component tang, loi selection nay xay ra nhieu lan hon.
- Combo reconstructed cuoi cung van duoc cham tren target tong, nhung tap ung vien dua vao combo da bi loc sai tu truoc.

## 3. Thay doi da thuc hien

### Sua actual component walk-forward

Da them `_make_actual_component_cache()` trong:

```text
src/vmd_ardl_ffnn/experiment_fixed.py
```

Logic moi:

- Tai moi validation/test date, lay `history + actual current row`.
- Chay VMD decomposition tren phan du lieu do.
- Lay gia tri component tai chinh date hien tai lam `actual_component`.
- Khong dung bat ky dong tuong lai nao.

Day la cach tao label component dung thang do ma van giu walk-forward/no-leakage.

### Sua output component prediction

`_predict_component_from_cache()` bay gio co the them cot:

```text
actual_component
```

vao `component_predictions_cached_vmd.csv` va `result["component_forecasts"]`.

Moi row component forecast co:

- `date`
- `split`
- `component`
- `predicted_component`
- `actual_component`
- `candidate_id`

### Sua scoring component

`_score_component_prediction()` da duoc sua tu:

```text
predicted_component vs actual_raw_transformed
```

thanh:

```text
predicted_component vs actual_component
```

Vi vay cac cot `Val RMSE`, `Val MAE`, `Val MAPE` trong component search gio phan anh dung loi cua component-level forecast.

### Giu nguyen reconstructed final metric

Phan final forecast van giu dung y tuong ban dau:

```text
predicted_reconstructed = sum(predicted_component theo date)
```

Sau do final metrics duoc tinh bang:

```text
predicted_reconstructed vs actual_raw_transformed
```

Tuc la:

- Component selection cham theo actual component.
- Combo/final forecast cham theo target tong.

## 4. Nhung thay doi lien quan activation/CLI da lam truoc do

Da them tuning activation giua:

```text
relu
tanh
```

qua `FFNNConfig.activation_grid`.

CLI da co:

```bash
--activation relu
--activation-grid relu tanh
--no-activation-tuning
```

Neu muon khong tune activation va chi dung mot activation co dinh:

```bash
python -m vmd_ardl_ffnn.cli \
  --data data.csv \
  --features X1 X2 \
  --activation tanh \
  --no-activation-tuning
```

## 5. Thay doi ve final metrics

Final metrics VMD/no-VMD da duoc them metadata:

- `dataset`
- `model`
- `uses_vmd`
- `vmd_setting`
- `vmd_modes`
- `activation`

Va khi ghi final metrics ra CSV se append thay vi overwrite.

Luu y: Neu doc file metrics cu trong `results/`, co the file da gom nhieu lan chay truoc/sau fix. Nen xem cot metadata va thoi diem output, hoac chay ra mot output folder moi de so sanh sach.

## 6. Test va kiem tra da them

Trong `tests/test_smoke.py` da them/bo sung:

- Kiem tra `component_forecasts` co cot `actual_component`.
- Kiem tra `actual_component` khong bi NaN.
- Kiem tra `component_count == modes + 1`.
- Regression test cho `_score_component_prediction()` de dam bao scorer dung `actual_component`, khong dung actual target tong.

Da chay duoc:

```bash
python -m compileall vmd_ardl_ffnn/src/vmd_ardl_ffnn vmd_ardl_ffnn/tests
git diff --check
```

Da chay smoke script synthetic cho VMD pipeline va ket qua:

- Co `actual_component`.
- Co du components `VMD1`, `VMD2`, `RES` khi `modes=2`.
- `component_count = 3`.
- Final metrics chay thanh cong.

Chua chay duoc full pytest vi moi truong hien tai thieu `pytest`:

```text
No module named pytest
```

## 7. Cach doc ket qua sau khi fix

Sau khi chay lai pipeline VMD:

1. Kiem tra `component_predictions_cached_vmd.csv`
   - Phai co cot `actual_component`.
   - Moi component nen co actual va predicted rieng.

2. Kiem tra `ffnn_component_validation_search_cached_vmd.csv`
   - `Val RMSE/MAE/MAPE` bay gio la metric component-level.
   - Khong nen so truc tiep cac metric nay voi final reconstructed metric.

3. Kiem tra `ffnn_validation_search_cached_vmd.csv`
   - Day la combo-level validation metric sau khi ghep components.
   - Cot nay moi la metric dung de chon combo reconstructed.

4. Kiem tra `final_reconstructed_metrics_cached_vmd.csv`
   - Day la final metric tren target tong.
   - Nen so sanh voi no-VMD bang cung split va cung scale.

## 8. Ky vong sau khi fix

Sau fix, viec tang `--vmd-modes` van co the lam sai so te hon neu:

- K qua cao lam tin hieu bi chia nho qua muc.
- So luong component nhieu lam tong loi du bao tich luy.
- Validation set ngan khong du de chon combo on dinh.
- Grid search/FFNN capacity chua phu hop voi component moi.

Nhung neu sai so te hon sau fix, do se la tin hieu model/du lieu that hon, khong con do bug chon candidate component bang metric sai thang do.

Khuyen nghi khi so sanh K:

- Chay vao output folder rieng cho tung K.
- Giu nguyen seed/grid/activation.
- So sanh final validation va test metric tren cung scale.
- Uu tien chon K theo validation reconstructed metric, khong chon theo component metric rieng le.
