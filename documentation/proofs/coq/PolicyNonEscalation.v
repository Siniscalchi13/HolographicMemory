From Coq Require Import List.
Import ListNotations.

Module Type LATTICE.
  Parameter L : Type.
  Parameter le : L -> L -> Prop.
  Infix "<=" := le : type_scope.
  Parameter join : L -> L -> L.
  Axiom le_refl : forall a, a <= a.
  Axiom le_trans : forall a b c, a <= b -> b <= c -> a <= c.
  Axiom le_antisym : forall a b, a <= b -> b <= a -> a = b.
  Axiom join_upper1 : forall a b, a <= join a b.
  Axiom join_upper2 : forall a b, b <= join a b.
  Axiom join_lub : forall a b c, a <= c -> b <= c -> join a b <= c.
End LATTICE.

Module NonEscalation (Lat : LATTICE).
  Import Lat.

  Fixpoint eff (base : L) (grants : list L) : L :=
    match grants with
    | [] => base
    | g :: gs => eff (join base g) gs
    end.

  Lemma eff_cons : forall base g gs,
      eff base (g::gs) = eff (join base g) gs.
  Proof. reflexivity. Qed.

  Theorem non_escalation_list : forall (cap base : L) (grs : list L),
      base <= cap ->
      (forall q, In q grs -> q <= cap) ->
      eff base grs <= cap.
  Proof.
    intros cap base grs Hbase Halls.
    induction grs as [|g gs IH].
    - simpl; assumption.
    - simpl. apply IH.
      + apply join_lub; [assumption|].
        apply Halls; simpl; auto.
      + intros q Hin. apply Halls. now right.
  Qed.

End NonEscalation.
