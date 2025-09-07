From Coq Require Import Reals List.
Import ListNotations.

Module QMC_Core.
  (* Abstract finite-dimensional inner-product setting over R for simplicity. *)
  Parameter V : Type.
  Parameter Rip : V -> V -> R.  (* inner product ⟨·,·⟩ *)

  Definition snorm (x:V) : R := Rip x x. (* squared norm *)

  (* Operators as endomorphisms *)
  Definition Op := V -> V.

  (* Unitary-as-isometry axiom: unitary operators preserve the inner product. *)
  Definition Unitary (U:Op) : Prop := forall x y, Rip (U x) (U y) = Rip x y.

  Theorem unitary_preserves_snorm : forall (U:Op), Unitary U -> forall psi, snorm (U psi) = snorm psi.
  Proof. intros U Hun psi. unfold snorm. now apply Hun. Qed.

  (* Orthonormal basis and Parseval (axiomatized) *)
  Parameter ONB : list V -> Prop.
  Parameter normalized : V -> Prop.

  (* Normalization: squared norm 1 *)
  Axiom normalized_def : forall psi, normalized psi <-> snorm psi = 1%R.

  Definition coeff (v psi:V) : R := Rip v psi.

  Fixpoint sumR (xs:list R) : R := match xs with | [] => 0%R | x::xt => x + sumR xt end.

  Axiom parseval : forall (B:list V) (psi:V),
      ONB B -> normalized psi ->
      sumR (map (fun v => (coeff v psi) * (coeff v psi)) B) = 1%R.

  Theorem projective_measurement_completeness : forall (B:list V) (psi:V),
      ONB B -> normalized psi ->
      sumR (map (fun v => (coeff v psi) * (coeff v psi)) B) = 1%R.
  Proof. intros. now apply parseval. Qed.

End QMC_Core.

