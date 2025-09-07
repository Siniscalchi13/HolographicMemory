From Coq Require Import Reals.

Module HMC_FFT.
  Parameter V : Type.
  Parameter Rip : V -> V -> R.
  Definition snorm (x:V) : R := Rip x x.
  Definition Op := V -> V.

  (* Unitary transform preserves inner product *)
  Definition Unitary (U:Op) : Prop := forall x y, Rip (U x) (U y) = Rip x y.

  Theorem unitary_preserves_snorm : forall (F:Op), Unitary F -> forall psi, snorm (F psi) = snorm psi.
  Proof. intros F Hun psi. unfold snorm; apply Hun. Qed.

  (* Interpreting FFT as a unitary operator in the appropriate normalization. *)
  Parameter FFT : Op.
  Axiom FFT_is_unitary : Unitary FFT.

  Corollary fft_preserves_snorm : forall psi, snorm (FFT psi) = snorm psi.
  Proof. intro psi. apply unitary_preserves_snorm; auto using FFT_is_unitary. Qed.

End HMC_FFT.

