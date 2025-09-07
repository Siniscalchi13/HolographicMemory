From Coq Require Import List Arith Lia.
From Coq Require Import Sorting.Sorted.
Import ListNotations.

Module EventOrder.

  Definition nat_lt := lt.

  Record Server := {
    log : list nat;
    log_sorted : StronglySorted nat_lt log
  }.

  Definition delivered_in_order (delivered:list nat) : Prop :=
    forall i j, i < j < length delivered -> nth i delivered 0 < nth j delivered 0.

  Definition subscribe (s:Server) : list nat := s.(log).

  Lemma nth_strict_increasing :
    forall (l:list nat) (i j:nat),
      StronglySorted lt l ->
      i < j < length l ->
      nth i l 0 < nth j l 0.
  Proof.
    induction l as [|a l' IH]; intros i j Hsorted [Hij Hlen]; simpl in *.
    - lia.
    - inversion Hsorted as [|? ? Hforall Hss]; subst.
      destruct i as [|i']; destruct j as [|j']; simpl in *.
      + lia.
      + (* i=0, j=S j' *)
        (* nth 0 (a::l') 0 = a; need a < nth j' l' 0 *)
        assert (j' < length l') by lia.
        clear Hij.
        (* From Forall (fun y => a < y) l' *)
        revert H; induction l' as [|b l'' IHl]; intros Hforall' Hjlen; simpl in *.
        * lia.
        * inversion Hforall' as [|b l'' Hab Hrest]; subst.
          destruct j' as [|j'']; simpl in *.
          { lia. }
          { apply IHl; assumption. }
      + (* i=S i', j=0 impossible *) lia.
      + (* i=S i', j=S j' *)
        assert (i' < j' < length l') by lia.
        specialize (IH i' j' Hss H). exact IH.
  Qed.

  Theorem subscribe_monotone : forall s,
      let d := subscribe s in
      delivered_in_order d.
  Proof.
    intros s d. unfold d, subscribe, delivered_in_order.
    intros i j Hij.
    apply nth_strict_increasing; [apply s.(log_sorted) | exact Hij].
  Qed.

End EventOrder.
